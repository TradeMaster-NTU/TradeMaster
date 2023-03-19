import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS

from trademaster.utils import get_attr, save_model, load_model, load_best_model, save_best_model, GeneralReplayBuffer
import numpy as np
import os
import pandas as pd
from collections import OrderedDict

@TRAINERS.register_module()
class OrderExecutionETEOTrainer(Trainer):
    def __init__(self, **kwargs):
        super(OrderExecutionETEOTrainer, self).__init__()
        self.num_envs = int(get_attr(kwargs, "num_envs", 1))

        self.device = get_attr(kwargs, "device", torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))

        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)

        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)

        self.num_threads = int(get_attr(kwargs, "num_threads", 8))
        self.time_steps = get_attr(kwargs, "time_steps", 10)

        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.if_discrete = get_attr(kwargs, "if_discrete", False)  # discrete or continuous action space
        self.if_off_policy = get_attr(kwargs, "if_off_policy", False)
        self.if_keep_save = get_attr(kwargs, "if_keep_save",
                                     True)  # keeping save the checkpoint. False means save until stop training.
        self.if_over_write = get_attr(kwargs, "if_over_write",
                                      False)  # overwrite the best policy network. `self.cwd/actor.pth`
        self.if_save_buffer = get_attr(kwargs, "if_save_buffer",
                                       False)  # if save the replay buffer for continuous training after stop training

        if self.if_off_policy:  # off-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 64))  # num of transitions sampled from replay buffer.
            self.horizon_len = int(
                get_attr(kwargs, "horizon_len", 512))  # collect horizon_len step while exploring, then update networks
            self.buffer_size = int(
                get_attr(kwargs, "buffer_size", 512))  # ReplayBuffer size. First in first out for off-policy.
        else:  # on-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 64))  # num of transitions sampled from replay buffer.
            self.horizon_len = int(
                get_attr(kwargs, "horizon_len", 2))  # collect horizon_len step while exploring, then update network
            self.buffer_size = int(
                get_attr(kwargs, "buffer_size", 128))  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.

        self.epochs = int(get_attr(kwargs, "epochs", 20))

        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim
        self.transition = self.agent.transition

        self.transition_shapes = OrderedDict({
            'state': (self.buffer_size, self.num_envs, self.time_steps, self.state_dim),
            'action': (self.buffer_size, self.num_envs, 2),
            'reward': (self.buffer_size, self.num_envs),
            'undone': (self.buffer_size, self.num_envs),
            'next_state': (self.buffer_size, self.num_envs, self.time_steps, self.state_dim),
        })

        self.init_before_training()

    def init_before_training(self):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            print(f"| Arguments Remove work_dir: {self.work_dir}")
        else:
            print(f"| Arguments Keep work_dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path, exist_ok=True)

    def train_and_valid(self):

        '''init agent.last_state'''
        state = self.agent.init_states(env=self.train_environment)
        if self.num_envs == 1:
            assert state.shape == (self.time_steps, self.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            assert state.shape == (self.num_envs, self.time_steps, self.state_dim)
            assert isinstance(state, torch.Tensor)
            state = state.to(self.device)
        assert state.shape == (self.num_envs, self.time_steps, self.state_dim)
        assert isinstance(state, torch.Tensor)
        self.agent.last_state = state.detach()

        '''init buffer'''
        if self.if_off_policy:
            buffer = GeneralReplayBuffer(transition=self.transition,
                                  shapes=self.transition_shapes,
                                  num_seqs=self.num_envs,
                                  max_size=self.buffer_size,
                                  device=self.device)
            buffer_items = self.agent.explore_env(env=self.train_environment, horizon_len=self.horizon_len)
            buffer.update(buffer_items)  # warm up for ReplayBuffer
        else:
            buffer = GeneralReplayBuffer(transition=self.transition,
                                         shapes=self.transition_shapes,
                                         num_seqs=self.num_envs,
                                         max_size=self.buffer_size,
                                         device=self.device)

        valid_score_list = []
        epoch = 1
        print("Train Episode: [{}/{}]".format(epoch, self.epochs))
        while True:
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len)

            buffer.update(buffer_items)

            torch.set_grad_enabled(True)
            logging_tuple = self.agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if torch.mean(buffer_items.undone) < 1.0:
                print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
                state = self.agent.init_states(env=self.valid_environment)
                if self.num_envs == 1:
                    state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    state = state.to(self.device)

                episode_reward_sum = 0.0  # sum of rewards in an episode
                while True:
                    action = self.agent.get_action(state.unsqueeze(0), if_train = False)  # different
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)

                    ary_action = action.detach().cpu().numpy()
                    ary_state, reward, done, _ = self.valid_environment.step(ary_action)  # next_state
                    ary_state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)

                    # compress state
                    state = torch.concat([state, ary_state.view(1, 1, -1)], dim=1)
                    state = state[:, 1:, :]

                    episode_reward_sum += reward
                    if done:
                        #print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                        break
                valid_score_list.append(self.valid_environment.cash_left)

                save_model(self.checkpoints_path,
                           epoch=epoch,
                           save=self.agent.get_save())
                epoch += 1
                if epoch <= self.epochs:
                    print("Train Episode: [{}/{}]".format(epoch, self.epochs))

            if epoch > self.epochs:
                break

        max_index = np.argmax(valid_score_list)
        load_model(self.checkpoints_path,
                   epoch=max_index + 1,
                   save=self.agent.get_save())
        save_best_model(
            output_dir=self.checkpoints_path,
            epoch=max_index + 1,
            save=self.agent.get_save()
        )

    def test(self):
        load_best_model(self.checkpoints_path, save=self.agent.get_save(), is_train=False)

        print("Test Best Episode")
        self.test_environment.reset()
        state = self.agent.init_states(env=self.test_environment)
        if self.num_envs == 1:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state = state.to(self.device)

        episode_reward_sum = 0
        while True:
            action = self.agent.get_action(state.unsqueeze(0), if_train=False)  # different
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, return_dict = self.test_environment.step(ary_action)  # next_state
            ary_state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)

            # compress state
            state = torch.concat([state, ary_state.view(1, 1, -1)], dim=1)
            state = state[:, 1:, :]

            episode_reward_sum += reward
            if done:
#                print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        result = np.array(self.test_environment.portfolio_value_history)
        np.save(os.path.join(self.work_dir,"result.npy"), result)
        return return_dict
