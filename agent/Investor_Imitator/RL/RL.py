import argparse
from re import A
from aiohttp import TraceResponseChunkReceivedParams
import numpy as np
import gym
import yaml
import torch.optim as optim
from torch.distributions import Categorical
import random
import os
import sys

sys.path.append("./")
from agent.Investor_Imitator.model.net import MLP_reg, MLP_cls
import torch
import pandas as pd
from agent.Investor_Imitator.logic_discriptor.utli import dict_to_args

parser = argparse.ArgumentParser()

parser.add_argument(
    "--env_name",
    choices=["portfolio"],
    default="portfolio",
    help="the name of TradingEnv ",
)
parser.add_argument(
    "--dict_trained_model",
    default="result/invest_imitator/trained_model",
    help="the dict of the trained model ",
)

parser.add_argument(
    "--train_env_config_dict",
    default=
    "config/input_config/env/portfolio/portfolio_for_investor_imitator/train.yml",
    help="the dict of the train config of TradingEnv ",
)

parser.add_argument(
    "--valid_env_config_dict",
    default=
    "config/input_config/env/portfolio/portfolio_for_investor_imitator/valid.yml",
    help="the dict of the valid config of TradingEnv ",
)

parser.add_argument(
    "--test_env_config_dict",
    default=
    "config/input_config/env/portfolio/portfolio_for_investor_imitator/test.yml",
    help="the dict of the test config of TradingEnv ",
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="the number of training epoch",
)

parser.add_argument(
    "--random_seed",
    type=int,
    default=12345,
    help="the number of training epoch",
)

parser.add_argument(
    "--result_dict",
    type=str,
    default="result/invest_imitator/test_result/",
    help="the dict of the result of the test",
)
parser.add_argument(
    "--input_config_RL",
    type=bool,
    default=False,
    help="whether to use yaml as the config",
)
parser.add_argument(
    "--input_config_dict_RL",
    type=str,
    default="config/input_config/agent/investor_imitator/RL.yml",
    help="the dict of yaml",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the random seed",
)


def env_creator(env_name):
    if env_name == 'portfolio':
        from env.PM.portfolio_for_investor_imitator import TradingEnv as env
    else:
        raise NotImplementedError
    return env


def load_yaml(dict):
    realpath = os.path.abspath(".")
    file_dict = os.path.join(realpath, dict)
    with open(file_dict, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


args = parser.parse_args()


class Agent:
    def __init__(self, args):
        if args.input_config_RL == True:
            with open(args.input_config_dict_RL, "r") as file:
                self.config = yaml.safe_load(file)
            args = dict_to_args(self.config)
        self.seed = args.seed
        self.set_seed()
        self.train_env_instance = env_creator(args.env_name)(load_yaml(
            args.train_env_config_dict))
        self.test_env_instance = env_creator(args.env_name)(load_yaml(
            args.test_env_config_dict))
        self.valid_env_instance = env_creator(args.env_name)(load_yaml(
            args.valid_env_config_dict))
        self.num_epoch = args.num_epochs
        self.policy=MLP_cls(input_size=self.train_env_instance.observation_space.shape[1],\
            hidden_size=128,output_size=self.train_env_instance.action_space.n).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.eps = np.finfo(np.float32).eps.item()
        self.eps = 1e-4
        self.result_dict = args.result_dict

        self.policy_path = args.dict_trained_model
        if not os.path.exists(self.policy_path):
            os.makedirs(self.policy_path)

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def select_action(self, state):
        state = torch.from_numpy(state).float().cuda()
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            # R = r + R * self.gamma
            R = r
            returns.insert(0, R)
        returns = torch.tensor(returns)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()  # 求和
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]  # 清空episode 数据
        del self.policy.saved_log_probs[:]

    def train_with_valid(self):
        rewards_list = []

        for i in range(self.num_epoch):
            # train
            state = self.train_env_instance.reset()
            done = False
            actions = []
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = self.train_env_instance.step(action)
                actions.append(action)
                self.policy.rewards.append(reward)

            self.finish_episode()
            model_path = self.policy_path + "/" + "all_model/" + "num_epoch_" + str(
                i + 1)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = model_path + "/" + "policy_gradient.pth"
            torch.save(self.policy, model_path)
            # valid
            state = self.valid_env_instance.reset()
            done = False
            rewards = 0
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = self.valid_env_instance.step(action)
                rewards += reward
            rewards_list.append(rewards)
        best_model_index = rewards_list.index(max(rewards_list))
        best_model_path = self.policy_path + "/" + "best_model/"
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        best_model_path = best_model_path + "/policy_gradient.pth"
        policy = torch.load(self.policy_path + "/" + "all_model/" +
                            "num_epoch_" + str(best_model_index + 1) +
                            "/policy_gradient.pth")
        torch.save(policy, best_model_path)
        self.best_model_path = best_model_path

    def test(self):
        self.policy = torch.load(self.best_model_path)
        state = self.test_env_instance.reset()
        done = False
        while not done:
            action = self.select_action(state)
            state, reward, done, _ = self.test_env_instance.step(action)
        rewards = self.test_env_instance.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_env_instance.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        if not os.path.exists(self.result_dict):
            os.makedirs(self.result_dict)
        df.to_csv(self.result_dict + "result.csv")
        return rewards


if __name__ == "__main__":
    agent = Agent(args)
    agent.train_with_valid()
    agent.test()
