from logging import raiseExceptions
from re import A, L
import sys
from turtle import done

sys.path.append(".")
from agent.ETEO.model import FCN_stack_ETTO, LSTM_ETEO
from agent.ETEO.util import set_seed, load_yaml
from env.OE.order_execution_for_ETEO import TradingEnv
import torch
import os
import argparse
from torch import nn
import random
from torch.distributions import Normal
import numpy as np
import pandas as pd
from random import sample

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed",
                    type=int,
                    default=12345,
                    help="the value of the random seed")
parser.add_argument("--env_config_path",
                    type=str,
                    default="config/input_config/env/OE/OE_for_ETEO/",
                    help="the path for storing the config file for deeptrader")
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="learning rate",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="result/ETEO/trained_model",
    help="the path for trained model",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/ETEO/test_result",
    help="the path for test result",
)
parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="the number of epoch we train",
)
parser.add_argument(
    "--net_category",
    type=str,
    default="stacked",
    choices=["stacked", "lstm"],
    help="the name of the category of the net we use for v and action",
)
parser.add_argument(
    "--lenth_state",
    type=int,
    default=10,
    help=
    "the length of the state, ie the number of timestamp that contains in the input of the net",
)
parser.add_argument(
    "--max_memory_capcity",
    type=int,
    default=1000,
    help=
    "the length of the state, ie the number of timestamp that contains in the input of the net",
)

parser.add_argument(
    "--sample_effiency",
    type=float,
    default=0.5,
    help="the portion of sample that could be used as material of ",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.9,
    help="the portion of sample that could be used as material of ",
)
parser.add_argument(
    "--climp",
    type=float,
    default=0.2,
    help="the value of climp ",
)


class trader:
    def __init__(self, args) -> None:
        self.seed = args.random_seed
        set_seed(self.seed)
        self.num_epoch = args.num_epoch
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cpu' if self.GPU_IN_USE else 'cpu')
        self.model_path = args.model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.result_path = args.result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.train_env_config = load_yaml(args.env_config_path + "train.yml")
        self.valid_env_config = load_yaml(args.env_config_path + "valid.yml")
        self.test_env_config = load_yaml(args.env_config_path + "test.yml")
        self.train_env_instance = TradingEnv(self.train_env_config)
        self.valid_env_instance = TradingEnv(self.valid_env_config)
        self.test_env_instance = TradingEnv(self.test_env_config)
        self.num_features = self.train_env_instance.observation_space.shape[0]
        self.net_category = args.net_category
        self.gamma = args.gamma
        self.climp = args.climp
        self.lenth_state = args.lenth_state
        # 两套网络（新与旧 来对比计算重采样的大小以及此次更新的大小尺度）
        # 由于两种网络的输入不同 因此我们这里目前只写stacked版本的更新
        if args.net_category not in ["stacked", "lstm"]:
            raiseExceptions(
                "we haven't implement that kind of net, please choose stacked or lstm"
            )
        if args.net_category == "stacked":
            self.net_old = FCN_stack_ETTO(args.lenth_state,
                                          self.num_features).to(self.device)
            self.net_new = FCN_stack_ETTO(args.lenth_state,
                                          self.num_features).to(self.device)
        if args.net_category == "lstm":
            self.net_old = LSTM_ETEO(args.lenth_state,
                                     self.num_features).to(self.device)
            self.net_new = LSTM_ETEO(args.lenth_state,
                                     self.num_features).to(self.device)
        self.max_memory_capcity = args.max_memory_capcity
        self.memory_size = 0
        #inputs: previous state(self.length的长度)
        #action：
        self.inputs = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.previous_rewards = []  # needed by lstm
        # 在后面训练 valid以及test时构建stacked states
        self.stacked_state = []
        self.dones = []
        self.sample_effiency = args.sample_effiency
        self.optimizer = torch.optim.Adam(self.net_new.parameters(),
                                          lr=args.lr)

    def compute_action(self, stacked_state):
        # stacked_state is a list of the previous state,(np.array with shape (156,)), whose length is 10
        list_states = []
        for state in stacked_state:
            state = torch.from_numpy(state).reshape(1, -1).float()
            list_states.append(state)
        list_states = torch.cat(list_states, dim=0).to(self.device)
        action_volume, action_price, v = self.net_old(list_states)
        action_volume = action_volume.squeeze()
        action_price = action_price.squeeze()
        v = v.squeeze(0)
        dis_volume = torch.distributions.normal.Normal(
            torch.relu(action_volume[0]) + 0.001,
            torch.relu(action_volume[1]) + 0.001)
        dis_price = torch.distributions.normal.Normal(
            torch.relu(action_price[0]) + 0.001,
            torch.relu(action_price[1]) + 0.001)
        volume = dis_volume.sample()
        price = dis_price.sample()
        action = np.array([torch.abs(volume).item(), torch.abs(price).item()])
        return action

    def compute_action_test(self, stacked_state):
        # stacked_state is a list of the previous state,(np.array with shape (156,)), whose length is 10
        list_states = []
        for state in stacked_state:
            state = torch.from_numpy(state).reshape(1, -1).float()
            list_states.append(state)
        list_states = torch.cat(list_states, dim=0).to(self.device)
        action_volume, action_price, v = self.net_old(list_states)
        action_volume = action_volume.squeeze()
        action_price = action_price.squeeze()
        v = v.squeeze(0)
        action = np.array([
            torch.relu(action_volume[0]).item() + 0.001,
            torch.relu(action_price[0]).item() + 0.001
        ])
        return action

    def save_transication(self, s, a, r, s_, r_previous, done):
        # here, the s,a,r,s_,r_previous are all torch tensor and in the GPU
        # self.memory_size = self.memory_size + 1
        if self.memory_size <= self.max_memory_capcity:
            self.inputs.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(s_)
            self.previous_rewards.append(r_previous)
            self.dones.append(done)
        else:
            index = self.memory_size % self.max_memory_capcity
            self.inputs[index - 1] = s
            self.actions[index - 1] = a
            self.rewards[index - 1] = r
            self.next_states[index - 1] = s_
            self.previous_rewards[index - 1] = r_previous
            self.dones[index - 1] = done

    def update(self):
        inputs = []
        actions = []
        rewards = []
        next_states = []
        previous_rewards = []
        dones = []
        number_sample = int(len(self.inputs) * self.sample_effiency)
        sample_list_number = sample(range(len(self.inputs)), number_sample)
        for i in sample_list_number:
            inputs.append(self.inputs[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            next_states.append(self.next_states[i])
            previous_rewards.append(self.previous_rewards[i])
            dones.append(self.dones[i])
        for input, action, reward, next_state, previous_reward, done in zip(
                inputs, actions, rewards, next_states, previous_rewards,
                dones):
            action_volume, action_price, v = self.net_old(next_state)

            td_target = reward + self.gamma * v * (1 - done)
            action_volume, action_price, v = self.net_old(input)
            action_volume, action_price, v = action_volume.squeeze(
            ), action_price.squeeze(), v.squeeze(0)
            mean = torch.cat(
                (action_volume[0].unsqueeze(0), action_price[0].unsqueeze(0)))
            std = torch.cat((torch.relu(action_volume[1].unsqueeze(0)) + 0.001,
                             torch.relu(action_price[1].unsqueeze(0)) + 0.001))
            old_dis = torch.distributions.normal.Normal(mean, std)
            log_prob_old = old_dis.log_prob(action).float()
            log_prob_old = (log_prob_old[0] + log_prob_old[1]).float()
            action_volume, action_price, v_s = self.net_new(next_state)
            action_volume, action_price, v = self.net_new(input)
            # td_error = torch.min(reward + self.gamma * v_s * (1 - done) - v,
            #                      torch.tensor([100]))
            td_error = reward + self.gamma * v_s * (1 - done) - v
            td_error = td_error.reshape(-1)

            # here is a little different from the original PPO, because there is a processure of passing the td error to different
            # state, however, we are only use 1 state at one time and do the update, therefore, we are simpling use the td error
            # we use td error instead of A to do the optimization
            action_volume, action_price, v = self.net_new(input)
            action_volume, action_price, v = action_volume.squeeze(
            ), action_price.squeeze(), v.squeeze(0)
            mean = torch.cat(
                (action_volume[0].unsqueeze(0), action_price[0].unsqueeze(0)))
            std = torch.cat((torch.relu(action_volume[1].unsqueeze(0)) + 0.001,
                             torch.relu(action_price[1].unsqueeze(0)) + 0.001))

            new_dis = torch.distributions.normal.Normal(mean, std)
            log_prob_new = new_dis.log_prob(action).float()
            log_prob_new = log_prob_new[0].float() + log_prob_new[1].float()

            ratio = torch.exp(
                torch.min(log_prob_new - log_prob_old, torch.tensor([10])))
            L1 = ratio * td_error.float()
            L2 = torch.clamp(ratio, 1 - self.climp,
                             1 + self.climp) * td_error.float()
            loss_pi = -torch.min(L1, L2).mean().float()
            # loss_pi = torch.min(loss_pi, torch.tensor([100000000]))
            loss_v = torch.min(
                torch.nn.functional.mse_loss(td_target.detach().reshape(-1),
                                             v.reshape(-1).float()),
                torch.tensor([1000000000]))
            loss_v = torch.nn.functional.mse_loss(
                td_target.detach().reshape(-1),
                v.reshape(-1).float())
            loss = loss_pi.float() + loss_v.float()
            loss.backward()
            self.optimizer.step()
        # self.net_old = self.net_new
        self.net_old.load_state_dict(self.net_new.state_dict(), strict=True)

    def train_with_valid(self):
        reward_list = []
        all_model_path = self.model_path + "/all_model/"
        best_model_path = self.model_path + "/best_model/"
        if not os.path.exists(all_model_path):
            os.makedirs(all_model_path)
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        for i in range(self.num_epoch):
            num_epoch = i
            stacked_state = []
            s = self.train_env_instance.reset()
            stacked_state.append(s)
            for i in range(self.lenth_state - 1):
                action = np.array([0, 0])
                s, r, done, _ = self.train_env_instance.step(action)
                stacked_state.append(s)
            action = self.compute_action(stacked_state)
            done = False
            i = 0
            while not done:
                i = i + 1
                old_states = []
                for state in stacked_state.copy():
                    state = torch.from_numpy(state).reshape(1, -1).float()
                    old_states.append(state)
                old_states = torch.cat(old_states, dim=0).float().to(a.device)
                action = self.compute_action(stacked_state)
                s_new, reward, done, _ = self.train_env_instance.step(action)
                stacked_state.pop(0)
                stacked_state.append(s_new)
                new_states = []
                for state in stacked_state.copy():
                    state = torch.from_numpy(state).reshape(1, -1).float()
                    new_states.append(state)
                new_states = torch.cat(new_states,
                                       dim=0).float().to(self.device)
                self.save_transication(
                    old_states,
                    torch.from_numpy(action).reshape(-1).float().to(
                        self.device),
                    torch.tensor(reward).float().reshape(-1).to(self.device),
                    new_states, 0,
                    torch.tensor(done).float().reshape(-1).to(self.device))
                if i % 100 == 1:
                    print("updating")
                    self.update()
                    self.inputs = []
                    self.actions = []
                    self.rewards = []
                    self.next_states = []
                    self.previous_rewards = []
                    self.dones = []
            torch.save(
                self.net_old, all_model_path +
                "policy_state_value_net_{}.pth".format(num_epoch))
            stacked_state = []
            s = self.valid_env_instance.reset()
            stacked_state.append(s)
            for i in range(self.lenth_state - 1):
                action = np.array([0, 0])
                s, r, done, _ = self.valid_env_instance.step(action)
                stacked_state.append(s)
            done = False
            while not done:
                action = self.compute_action_test(stacked_state)
                s_new, reward, done, _ = self.valid_env_instance.step(action)
                stacked_state.pop(0)
                stacked_state.append(s_new)
            reward_list.append(reward)
        max_reward = max(reward_list)
        index = reward_list.index(max_reward)
        net_path = all_model_path + "policy_state_value_net_{}.pth".format(
            index)
        self.net_old = torch.load(net_path)
        torch.save(self.net_old,
                   best_model_path + "policy_state_value_net.pth")

    def test(self):
        stacked_state = []
        s = self.test_env_instance.reset()
        stacked_state.append(s)
        for i in range(self.lenth_state - 1):
            action = np.array([0, 0])
            s, r, done, _ = self.test_env_instance.step(action)
            stacked_state.append(s)
        done = False
        while not done:
            action = self.compute_action_test(stacked_state)
            s_new, reward, done, _ = self.test_env_instance.step(action)
            stacked_state.pop(0)
            stacked_state.append(s_new)
        result = np.array(self.test_env_instance.portfolio_value_history)
        np.save(self.result_path + "/result.npy", result)


if __name__ == "__main__":
    args = parser.parse_args()
    a = trader(args)
    a.train_with_valid()
    a.test()
