from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_model, save_best_model, load_model, load_best_model
import os
import pandas as pd
import random
import numpy as np
from collections import OrderedDict

@TRAINERS.register_module()
class PortfolioManagementInvestorImitatorTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementInvestorImitatorTrainer, self).__init__()

        self.kwargs = kwargs
        self.device = get_attr(kwargs, "device", None)
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)
        self.num_threads = int(get_attr(kwargs, "num_threads", 8))

        self.work_dir = os.path.join(ROOT, self.work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

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

        valid_score_list = []
        for epoch in range(1, self.epochs + 1):
            print("Train Episode: [{}/{}]".format(epoch, self.epochs))
            state = self.train_environment.reset()

            actions = []
            episode_reward_sum = 0
            while True:
                action = self.agent.get_action(state)
                state, reward, done, _ = self.train_environment.step(action)
                actions.append(action)
                episode_reward_sum += reward
                self.agent.act.rewards.append(reward)
                if done:
                    # print("Train Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break

            self.agent.update_net()

            save_model(self.checkpoints_path,
                       epoch=epoch,
                       save=self.agent.get_save())

            print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
            state = self.valid_environment.reset()

            episode_reward_sum = 0
            while True:
                action = self.agent.get_action(state)
                state, reward, done, _ = self.valid_environment.step(action)
                episode_reward_sum += reward
                if done:
                    #print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break
            valid_score_list.append(episode_reward_sum)

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

        state = self.test_environment.reset()
        episode_reward_sum = 0
        while True:
            action = self.agent.get_action(state)
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
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"))
        return daily_return

    def test_with_customize_policy(self, policy, customize_policy_id, extra_parameters=None):

        self.test_environment.test_id = customize_policy_id
        print(f"Test customize policy: {str(customize_policy_id)}")
        state = self.test_environment.reset()
        episode_reward_sum = 0
        weights_brandnew=None
        while True:


            if customize_policy_id=="Average_holding":
                weights = policy(state, self.test_environment,weights_brandnew)
            else:
                weights = policy(state, self.test_environment)


            state, reward, done, return_dict = self.test_environment.step(None,weights)
            episode_reward_sum += reward
            if done:
                # print("Test customize policy Reward Sum: {:04f}".format(episode_reward_sum))
                break
            weights_brandnew = return_dict["weights_brandnew"]
        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"))
        return daily_return
