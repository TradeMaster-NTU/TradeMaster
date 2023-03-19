from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr
import numpy as np
import os
import pandas as pd
import random
from collections import OrderedDict, namedtuple
from trademaster.utils import get_attr, save_model, load_model, load_best_model, save_best_model, ReplayBuffer, GeneralReplayBuffer


@TRAINERS.register_module()
class OrderExecutionPDTrainer(Trainer):
    def __init__(self, **kwargs):
        super(OrderExecutionPDTrainer, self).__init__()
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
        self.if_off_policy = get_attr(kwargs, "if_off_policy", True)
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
                get_attr(kwargs, "horizon_len", 512))  # collect horizon_len step while exploring, then update network
            self.buffer_size = int(
                get_attr(kwargs, "buffer_size", 512))  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.

        self.epochs = int(get_attr(kwargs, "epochs", 20))

        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim

        self.public_state_dim = self.agent.public_state_dim
        self.private_state_dim = self.agent.private_state_dim
        self.time_steps = get_attr(kwargs, "time_steps", 10)
        self.batch_size = int(get_attr(kwargs, "batch_size", 64))

        self.transition = self.agent.transition
        self.transition_shapes = OrderedDict({
            'state': (self.buffer_size, self.num_envs, self.time_steps, self.state_dim),
            'action': (self.buffer_size, self.num_envs, 1),
            'reward': (self.buffer_size, self.num_envs),
            'undone': (self.buffer_size, self.num_envs),
            'next_state': (self.buffer_size, self.num_envs, self.time_steps, self.state_dim),
            'public_state':(self.buffer_size, self.num_envs, self.time_steps * 2, self.public_state_dim),
            'private_state':(self.buffer_size, self.num_envs, self.time_steps, self.private_state_dim),
            'next_public_state':(self.buffer_size, self.num_envs, self.time_steps * 2, self.public_state_dim),
            'next_private_state':(self.buffer_size, self.num_envs, self.time_steps, self.private_state_dim),
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
        state, info = self.train_environment.reset()
        public_state = info["perfect_state"]
        private_state = info["private_state"]

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        public_state = torch.tensor(public_state, dtype=torch.float32, device=self.device)
        private_state = torch.tensor(private_state, dtype=torch.float32, device=self.device)

        self.agent.last_state = state
        self.agent.last_public_state = public_state
        self.agent.last_private_state = private_state

        '''init buffer'''
        if self.if_off_policy:
            teacher_buffer = GeneralReplayBuffer(transition=self.transition,
                                         shapes=self.transition_shapes,
                                         num_seqs=self.num_envs,
                                         max_size=self.buffer_size,
                                         device=self.device)
            buffer_items = self.agent.explore_env(env=self.train_environment, horizon_len=self.horizon_len, if_teacher=True)
            teacher_buffer.update(buffer_items)
            student_buffer = GeneralReplayBuffer(transition=self.transition,
                                         shapes=self.transition_shapes,
                                         num_seqs=self.num_envs,
                                         max_size=self.buffer_size,
                                         device=self.device)
        else:
            teacher_buffer = GeneralReplayBuffer(transition=self.transition,
                                                 shapes=self.transition_shapes,
                                                 num_seqs=self.num_envs,
                                                 max_size=self.buffer_size,
                                                 device=self.device)
            student_buffer = GeneralReplayBuffer(transition=self.transition,
                                                 shapes=self.transition_shapes,
                                                 num_seqs=self.num_envs,
                                                 max_size=self.buffer_size,
                                                 device=self.device)

        valid_score_list = []
        for epoch in range(1, self.epochs+1):
            print("Train Episode: [{}/{}]".format(epoch, self.epochs))
            # train teacher
            while True:
                buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len, if_teacher=True)

                teacher_buffer.update(buffer_items)

                torch.set_grad_enabled(True)
                logging_tuple = self.agent.update_teacher(teacher_buffer)
                torch.set_grad_enabled(False)

                if torch.mean(buffer_items.undone) < 1.0:
                    break

            state, info = self.train_environment.reset()
            public_state = info["perfect_state"]
            private_state = info["private_state"]

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            public_state = torch.tensor(public_state, dtype=torch.float32, device=self.device)
            private_state = torch.tensor(private_state, dtype=torch.float32, device=self.device)

            self.agent.last_state = state
            self.agent.last_public_state = public_state
            self.agent.last_private_state = private_state

            # train student
            while True:
                buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len, if_teacher = False)

                student_buffer.update(buffer_items)

                torch.set_grad_enabled(True)
                logging_tuple = self.agent.update_student(student_buffer)
                torch.set_grad_enabled(False)

                if torch.mean(buffer_items.undone) < 1.0:
                    print("Valid Episode: [{}/{}]".format(epoch, self.epochs))

                    state, info = self.valid_environment.reset()

                    episode_reward_sum = 0.0  # sum of rewards in an episode
                    while True:
                        public_state = torch.from_numpy(state).to(self.device).float()
                        private_state = torch.from_numpy(info["private_state"]).to(self.device).float()
                        tensor_action = self.agent.get_student_action(public_state, private_state)
                        action = tensor_action[0, 0].detach().cpu().numpy()
                        state, reward, done, info = self.valid_environment.step(action)
                        episode_reward_sum += reward
                        if done:
                            #print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                            break
                    valid_score_list.append(episode_reward_sum)
                    save_model(self.checkpoints_path,
                               epoch=epoch,
                               save=self.agent.get_save())
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

        state, info = self.test_environment.reset()

        episode_reward_sum = 0
        while True:

            public_state = torch.from_numpy(state).to(self.device).float()
            private_state = torch.from_numpy(info["private_state"]).to(self.device).float()

            tensor_action = self.agent.get_student_action(public_state, private_state)
            action = tensor_action[0, 0].detach().cpu().numpy()
            state, reward, done, info = self.test_environment.step(action)
            episode_reward_sum += reward

            if done:
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break
        return info