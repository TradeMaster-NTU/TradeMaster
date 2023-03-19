import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_model,load_model, load_best_model, save_best_model, ReplayBufferHFT
import numpy as np
import os
import pandas as pd
@TRAINERS.register_module()
class HighFrequencyTradingTrainer(Trainer):
    def __init__(self, **kwargs):
        super(HighFrequencyTradingTrainer, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))

        self.device = get_attr(
            kwargs, "device",
            torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))

        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)

        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)

        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)

        self.epochs = int(get_attr(kwargs, "epochs", 2))

        self.num_threads = int(get_attr(kwargs, "num_threads", 8))

        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.if_discrete = get_attr(
            kwargs, "if_discrete",
            False)  # discrete or continuous action space
        self.if_off_policy = get_attr(kwargs, "if_off_policy", True)
        self.if_keep_save = get_attr(
            kwargs, "if_keep_save", True
        )  # keeping save the checkpoint. False means save until stop training.
        self.if_over_write = get_attr(
            kwargs, "if_over_write",
            False)  # overwrite the best policy network. `self.cwd/actor.pth`
        self.if_save_buffer = get_attr(
            kwargs, "if_save_buffer", False
        )  # if save the replay buffer for continuous training after stop training

        if self.if_off_policy:  # off-policy
            self.batch_size = int(get_attr(
                kwargs, "batch_size",
                512))  # num of transitions sampled from replay buffer.
            self.horizon_len = int(
                get_attr(kwargs, "horizon_len", 512)
            )  # collect horizon_len step while exploring, then update networks
            self.buffer_size = int(get_attr(
                kwargs, "buffer_size",
                1e5))  # ReplayBuffer size. First in first out for off-policy.
            self.n_step = get_attr(kwargs, "n_step", 10)

        else:  # on-policy
            raise Exception("DDQN is a off-line RL algorithms")
        self.agent = get_attr(kwargs, "agent", None)
        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim

        self.init_before_training()

        self.random_start_list = random.sample(
            range(
                len(self.train_environment.df) -
                self.train_environment.episode_length),
            self.epochs * int(
                len(self.train_environment.df) /
                self.train_environment.episode_length))

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
            self.if_remove = bool(
                input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") ==
                'y')
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
        if self.if_off_policy:
            buffer = ReplayBufferHFT(
                device=self.device,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                seed=self.random_seed,
                gamma=self.agent.gamma,
                n_step=self.n_step,
            )
            buffer = self.agent.explore_env(self.train_environment,
                                            self.horizon_len,
                                            buffer,
                                            self.random_start_list,
                                            if_random=True)
        else:
            raise Exception("DDQN is a off-line RL algorithms")

        valid_score_list = []
        epoch = 1
        print("Train Episode: [{}/{}]".format(epoch, self.epochs))
        while True:
            previous_completed_eposide_number=self.agent.completed_eposide_number
            buffer = self.agent.explore_env(self.train_environment,
                                            self.horizon_len,
                                            buffer,
                                            self.random_start_list,
                                            if_random=False)

            torch.set_grad_enabled(True)
            logging_tuple = self.agent.update_net(buffer)
            torch.set_grad_enabled(False)
            current_completed_eposide_number=self.agent.completed_eposide_number

            if current_completed_eposide_number!=previous_completed_eposide_number and (self.agent.completed_eposide_number % (
                int(len(self.train_environment.df) /
                 self.train_environment.episode_length)) == (
                     int(len(self.train_environment.df) /
                      self.train_environment.episode_length) - 1)):
                #如果环境中出现undone的情况
                print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
                state, info = self.valid_environment.reset()
                episode_reward_sum = 0.0  # sum of rewards in an episode
                while True:
                    action = self.agent.get_action_test(state, info)
                    state, reward, done, info = self.valid_environment.step(
                        action)
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
        save_best_model(output_dir=self.checkpoints_path,
                        epoch=max_index + 1,
                        save=self.agent.get_save())

    def test(self):
        load_best_model(self.checkpoints_path,
                        save=self.agent.get_save(),
                        is_train=False)

        print("Test Best Episode")
        state, info = self.test_environment.reset()
        while True:
            action = self.agent.get_action_test(state, info)
            state, reward, done, info = self.test_environment.step(action)
            if done:
                break
        df = self.test_environment.save_asset_memoey()
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
        daily_return = df.daily_return.values
        return daily_return

    def test_with_customize_policy(self, policy, customize_policy_id):

        state, info = self.test_environment.reset()
        self.test_environment.test_id = customize_policy_id
        print(f"Test customize policy: {str(customize_policy_id)}")
        while True:
            action = policy(state, self.test_environment)
            # print(action)
            action = np.int64(action)
            state, reward, done, info = self.test_environment.step(action)
            if done:
                break
        df = self.test_environment.save_asset_memoey()
        df.to_csv(os.path.join(self.work_dir, 'test_result_customize_policy_'+str(customize_policy_id)+'.csv'), index=False)
        daily_return = df.daily_return.values
        return daily_return