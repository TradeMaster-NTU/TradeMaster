import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_model, load_model, load_best_model, save_best_model, save_best_model_trial, ReplayBuffer, GeneralReplayBuffer
import numpy as np
import os
import pandas as pd
from collections import namedtuple, OrderedDict
"""the Algorithm trading DeepScalper is based on the paper  'DeepScalper: A Risk-Aware Reinforcement Learning Framework
to Capture Fleeting Intraday Trading Opportunities'(https://arxiv.org/pdf/2201.09058.pdf)"""

@TRAINERS.register_module()
class AlgorithmicTradingTrainer(Trainer):
    def __init__(self, **kwargs):
        super(AlgorithmicTradingTrainer, self).__init__()

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

        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.if_discrete = get_attr(kwargs, "if_discrete", False)
        self.if_off_policy = get_attr(kwargs, "if_off_policy", True)
        self.if_keep_save = get_attr(kwargs, "if_keep_save", True)
        self.if_over_write = get_attr(kwargs, "if_over_write", False)
        self.if_save_buffer = get_attr(kwargs, "if_save_buffer", False)

        if self.if_off_policy:  # off-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 64))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 1e6))
        else:  # on-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 128))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 128))
        self.epochs = int(get_attr(kwargs, "epochs", 20))

        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim
        self.transition = self.agent.transition

        self.transition_shapes = OrderedDict({
            'state':(self.buffer_size, self.num_envs, self.state_dim),
            'action': (self.buffer_size, self.num_envs, 1),
            'reward': (self.buffer_size, self.num_envs),
            'undone': (self.buffer_size, self.num_envs),
            'next_state':(self.buffer_size, self.num_envs, self.state_dim),
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
        state = self.train_environment.reset()
        if self.num_envs == 1:
            assert state.shape == (self.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            assert state.shape == (self.num_envs, self.state_dim)
            assert isinstance(state, torch.Tensor)
            state = state.to(self.device)
        assert state.shape == (self.num_envs, self.state_dim)
        assert isinstance(state, torch.Tensor)
        self.agent.last_state = state.detach()

        '''init buffer'''
        if self.if_off_policy:
            buffer = GeneralReplayBuffer(
                                  transition=self.transition,
                                  shapes=self.transition_shapes,
                                  num_seqs=self.num_envs,
                                  max_size=self.buffer_size,
                                  device=self.device,
                                  )
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len, if_random=True)
            buffer.update(buffer_items)
        else:
            buffer = []

        valid_score_list = []
        epoch = 1
        print("Train Episode: [{}/{}]".format(epoch, self.epochs))
        while True:
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len)
            if self.if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = self.agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if torch.mean(buffer_items.undone) < 1.0:
                print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
                state = self.valid_environment.reset()
                episode_reward_sum = 0.0  # sum of rewards in an episode
                while True:
                    tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    tensor_action = self.agent.get_action(tensor_state)
                    if self.if_discrete:
                        tensor_action = tensor_action.argmax(dim=1)
                    action = tensor_action.detach().cpu().numpy()[
                        0]  # not need detach(), because using torch.no_grad() outside
                    state, reward, done, _ = self.valid_environment.step(action)
                    episode_reward_sum += reward
                    if done:
                        #print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                        break
                valid_score_list.append(episode_reward_sum)

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
    
    def train_and_valid_trial(self, trial_number):

        '''init agent.last_state'''
        state = self.train_environment.reset()
        if self.num_envs == 1:
            assert state.shape == (self.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            assert state.shape == (self.num_envs, self.state_dim)
            assert isinstance(state, torch.Tensor)
            state = state.to(self.device)
        assert state.shape == (self.num_envs, self.state_dim)
        assert isinstance(state, torch.Tensor)
        self.agent.last_state = state.detach()

        '''init buffer'''
        if self.if_off_policy:
            buffer = GeneralReplayBuffer(
                                  transition=self.transition,
                                  shapes=self.transition_shapes,
                                  num_seqs=self.num_envs,
                                  max_size=self.buffer_size,
                                  device=self.device,
                                  )
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len, if_random=True)
            buffer.update(buffer_items)
        else:
            buffer = []

        valid_score_list = []
        epoch = 1
        print("Train Episode: [{}/{}]".format(epoch, self.epochs))
        while True:
            buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len)
            if self.if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = self.agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if torch.mean(buffer_items.undone) < 1.0:
                print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
                state = self.valid_environment.reset()
                episode_reward_sum = 0.0  # sum of rewards in an episode
                while True:
                    tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    tensor_action = self.agent.get_action(tensor_state)
                    if self.if_discrete:
                        tensor_action = tensor_action.argmax(dim=1)
                    action = tensor_action.detach().cpu().numpy()[
                        0]  # not need detach(), because using torch.no_grad() outside
                    state, reward, done, _ = self.valid_environment.step(action)
                    episode_reward_sum += reward
                    if done:
                        #print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                        break
                valid_score_list.append(episode_reward_sum)

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
                   epoch = max_index + 1,
                   save=self.agent.get_save())
        save_best_model_trial(
            output_dir=self.checkpoints_path,
            epoch=max_index + 1,
            trial_number=trial_number,
            save=self.agent.get_save()
        )
        return np.max(valid_score_list)

    def test(self):

        load_best_model(self.checkpoints_path, save=self.agent.get_save(), is_train=False)

        print("Test Best Episode")
        state = self.test_environment.reset()

        episode_reward_sum = 0
        while True:
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            tensor_action = self.agent.act(tensor_state)
            if self.if_discrete:
                tensor_action = tensor_action.argmax(dim=1)
            action = tensor_action.detach().cpu().numpy()[
                0]  # not need detach(), because using torch.no_grad() outside
            state, reward, done, _ = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
        return daily_return

    def test_with_customize_policy(self,policy,customize_policy_id):

        state = self.test_environment.reset()
        self.test_environment.test_id=customize_policy_id
        print(f"Test customize policy: {str(customize_policy_id)}")

        episode_reward_sum = 0

        while True:
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = policy(tensor_state,self.test_environment)
            # print(action)
            action=np.int64(action)
            # print('action is: ',action,type(action))
            state, reward, done, _ = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, 'test_result_customize_policy_'+str(customize_policy_id)+'.csv'), index=False)
        return daily_return

    def test_with_customize_actions(self,customize_actions,customize_actions_id):


        print(f"Test customize policy: {str(customize_actions_id)}")
        self.test_environment.test_id = customize_actions_id
        state = self.test_environment.reset()

        if len(customize_actions)!=self.test_environment.action_length:
            raise ValueError('Action length doesn\'t fit.')

        episode_reward_sum = 0
        action_index=0
        while True:
            action=customize_actions[action_index]
            action_index+=1
            # print('action is: ',action,type(action))
            if action<0 or action>self.test_environment.action_dim-1:
                raise ValueError('Action volume doesn\'t fit.')
            action=np.int64(action)
            state, reward, done, _ = self.test_environment.step(action)
            episode_reward_sum += reward
            if action_index==self.test_environment.action_length+1:
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result_customize_actions_id_"+str(customize_actions_id)+".csv"), index=False)
        return daily_return



