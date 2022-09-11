import sys

sys.path.append(".")
from agent.DeepTrader.data.market_information import make_market_information, make_correlation_information
from agent.DeepTrader.model.model import Chomp1d, TemporalBlock, TemporalConvNet, SA, GCN, IN, IN_value, asset_scoring, asset_scoring_value, market_scoring
from agent.DeepTrader.model.portfolio_generator import generate_portfolio, generate_rho
from agent.DeepTrader.util.utli import set_seed, load_yaml
from env.PM.portfolio_for_deeptrader import Tradingenv
import torch
import os
import argparse
from torch import nn
import random
from torch.distributions import Normal
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed",
                    type=int,
                    default=12345,
                    help="the value of the random seed")
parser.add_argument(
    "--env_config_path",
    type=str,
    default="config/input_config/env/portfolio/portfolio_for_deeptrader/",
    help="the path for storing the config file for deeptrader")

parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="the gamma for DPG",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="learning rate",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="result/DeepTrader/trained_model",
    help="the path for trained model",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DeepTrader/test_result",
    help="the path for test result",
)
parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="the number of epoch we train",
)
parser.add_argument(
    "--technical_indicator",
    type=list,
    default=[
        "high", "low", "open", "close", "adjcp", "zopen", "zhigh", "zlow",
        "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25", "zd_30"
    ],
    help="technical_indicator",
)


class trader:
    def __init__(self, args) -> None:
        self.lr = args.lr
        self.technical_indicator = args.technical_indicator
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
        self.train_env_instance = Tradingenv(self.train_env_config)
        self.valid_env_instance = Tradingenv(self.valid_env_config)
        self.test_env_instance = Tradingenv(self.test_env_config)
        self.day_length = self.train_env_config["length_day"]
        self.input_channel = len(self.train_env_config["tech_indicator_list"])
        self.asset_policy = asset_scoring(N=self.train_env_instance.stock_dim,
                                          K_l=self.day_length,
                                          num_inputs=self.input_channel,
                                          num_channels=[12, 12, 12])
        self.asset_policy_critic = asset_scoring_value(
            N=self.train_env_instance.stock_dim,
            K_l=self.day_length,
            num_inputs=self.input_channel,
            num_channels=[12, 12, 12])
        self.market_policy = market_scoring(self.input_channel)
        self.optimizer_asset_actor = torch.optim.Adam(
            self.asset_policy.parameters(), lr=self.lr)
        self.optimizer_asset_critic = torch.optim.Adam(
            self.asset_policy_critic.parameters(), lr=self.lr)
        self.optimizer_market_policy = torch.optim.Adam(
            self.market_policy.parameters(), lr=self.lr)
        self.memory_counter = 0
        self.memory_capacity = 1000

        self.s_memory_asset = []
        self.a_memory_asset = []
        self.r_memory_asset = []
        self.sn_memory_asset = []
        self.correlation_matrix = []
        self.correlation_n_matrix = []

        self.s_memory_market = []
        self.a_memory_market = []
        self.r_memory_market = []
        self.sn_memory_market = []
        self.roh_bars = []

        self.policy_update_frequency = 500
        self.critic_learn_time = 0
        self.gamma = args.gamma
        self.mse_loss = nn.MSELoss()

        self.asset_policy = self.asset_policy.to(self.device)
        self.asset_policy_critic = self.asset_policy_critic.to(self.device)
        self.market_policy = self.market_policy.to(self.device)

    def store_transition(self, s_asset, a_asset, r, sn_asset, s_market,
                         a_market, sn_market, A, A_n,
                         roh_bar):  # 定义记忆存储函数 (这里输入为两套transition：asset和market)

        self.memory_counter = self.memory_counter + 1
        if self.memory_counter < self.memory_capacity:
            self.s_memory_asset.append(s_asset)
            self.a_memory_asset.append(a_asset)
            self.r_memory_asset.append(r)
            self.sn_memory_asset.append(sn_asset)
            self.correlation_matrix.append(A)
            self.correlation_n_matrix.append(A_n)

            self.s_memory_market.append(s_market)
            self.a_memory_market.append(a_market)
            self.r_memory_market.append(r)
            self.sn_memory_market.append(sn_market)
            self.roh_bars.append(roh_bar)

        else:
            number = self.memory_counter % self.memory_capacity
            self.s_memory_asset[number - 1] = s_asset
            self.a_memory_asset[number - 1] = a_asset
            self.r_memory_asset[number - 1] = r
            self.sn_memory_asset[number - 1] = sn_asset
            self.correlation_matrix[number - 1] = A
            self.correlation_n_matrix[number - 1] = A_n

            self.s_memory_market[number - 1] = s_market
            self.a_memory_market[number - 1] = a_market
            self.r_memory_market[number - 1] = r
            self.sn_memory_market[number - 1] = sn_market
            self.roh_bars[number - 1] = roh_bar

    def compute_weights_train(self, asset_state, market_state, A):
        # random sample when compute roh
        asset_state = torch.from_numpy(asset_state).float().to(self.device)
        asset_scores = self.asset_policy(asset_state, A)
        input_market = torch.from_numpy(market_state).unsqueeze(0).to(
            torch.float32).to(self.device)
        output_market = self.market_policy(input_market)
        roh_bar = generate_rho(output_market[0].cpu(), output_market[1].cpu())
        normal = Normal(output_market[0].cpu(), output_market[1].cpu())
        self.roh_bar = roh_bar
        weights = generate_portfolio(asset_scores.cpu(), roh_bar)
        weights = weights.numpy()
        return weights

    def compute_weights_test(self, asset_state, market_state, A):
        # use the mean to compute roh
        asset_state = torch.from_numpy(asset_state).float().to(self.device)
        asset_scores = self.asset_policy(asset_state, A)
        input_market = torch.from_numpy(market_state).unsqueeze(0).to(
            torch.float32).to(self.device)
        output_market = self.market_policy(input_market)
        weights = generate_portfolio(asset_scores.cpu().detach(),
                                     output_market[0].cpu().detach().numpy())
        weights = weights.detach().numpy()
        return weights

    def learn(self):
        print("learning")
        length = len(self.s_memory_asset)
        out1 = random.sample(range(length), int(length / 10))
        # random sample
        s_learn_asset = []
        a_learn_asset = []
        r_learn_asset = []
        sn_learn_asset = []
        correlation_asset = []
        correlation_asset_n = []

        s_learn_market = []
        a_learn_market = []
        r_learn_market = []
        sn_learn_market = []
        roh_bar_market = []
        for number in out1:
            s_learn_asset.append(self.s_memory_asset[number])
            a_learn_asset.append(self.a_memory_asset[number])
            r_learn_asset.append(self.r_memory_asset[number])
            sn_learn_asset.append(self.sn_memory_asset[number])
            correlation_asset.append(self.correlation_matrix[number])
            correlation_asset_n.append(self.correlation_n_matrix[number])

            s_learn_market.append(self.s_memory_market[number])
            a_learn_market.append(self.a_memory_market[number])
            r_learn_market.append(self.r_memory_market[number])
            sn_learn_market.append(self.sn_memory_market[number])
            roh_bar_market.append(self.roh_bars[number])
        self.critic_learn_time = self.critic_learn_time + 1
        #update the asset unit
        # 除了correlation以外都是tensor correlation是np.array 直接从make_correlation_information返回即可
        print("update asset unit")
        for bs, ba, br, bs_, correlation, correlation_n in zip(
                s_learn_asset, a_learn_asset, r_learn_asset, sn_learn_asset,
                correlation_asset, correlation_asset_n):
            #update actor
            a = self.asset_policy(bs, correlation)
            q = self.asset_policy_critic(bs, correlation, a)
            a_loss = -torch.mean(q)
            self.optimizer_asset_actor.zero_grad()
            a_loss.backward(retain_graph=True)
            self.optimizer_asset_actor.step()
            #update critic
            a_ = self.asset_policy(bs_, correlation_n)
            q_ = self.asset_policy_critic(bs_, correlation_n, a_.detach())
            q_target = br + self.gamma * q_
            q_eval = self.asset_policy_critic(bs, correlation, ba.detach())
            # print(q_eval)
            # print(q_target)
            td_error = self.mse_loss(q_target.detach(), q_eval)
            # print(td_error)
            self.optimizer_asset_critic.zero_grad()
            td_error.backward()
            self.optimizer_asset_critic.step()
        #update the asset unit
        # 除了correlation以外都是tensor correlation是np.array 直接从make_correlation_information返回即可
        print("update market unit")
        loss_market = 0
        for s, br, roh_bar in zip(s_learn_market, r_learn_asset,
                                  roh_bar_market):
            output_market = self.market_policy(s)
            normal = Normal(output_market[0], output_market[1])
            b_prob = -normal.log_prob(roh_bar)

            loss_market += br * b_prob
        loss_market.backward()
        self.optimizer_market_policy.step()

    def train_with_valid(self):

        rewards_list = []
        for i in range(self.num_epoch):
            print("traning")
            j = 0
            done = False
            s = self.train_env_instance.reset()
            while not done:
                old_asset_state = s
                old_market_state = torch.from_numpy(
                    make_market_information(
                        self.train_env_instance.data,
                        technical_indicator=self.technical_indicator)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_old = make_correlation_information(
                    self.train_env_instance.data)
                weights = self.compute_weights_train(
                    s,
                    make_market_information(
                        self.train_env_instance.data,
                        technical_indicator=self.technical_indicator),
                    corr_matrix_old)
                action_asset = self.asset_policy(
                    torch.from_numpy(old_asset_state).float().to(self.device),
                    corr_matrix_old)
                action_market = self.market_policy(old_market_state)
                s, reward, done, _ = self.train_env_instance.step(weights)
                new_asset_state = s
                new_market_state = torch.from_numpy(
                    make_market_information(
                        self.train_env_instance.data,
                        technical_indicator=self.technical_indicator)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_new = make_correlation_information(
                    self.train_env_instance.data)
                self.store_transition(
                    torch.from_numpy(old_asset_state).float().to(self.device),
                    action_asset,
                    torch.tensor(reward).float().to(self.device),
                    torch.from_numpy(new_asset_state).float().to(self.device),
                    old_market_state, action_market, new_market_state,
                    corr_matrix_old, corr_matrix_new, self.roh_bar)
                j = j + 1
                if j % 100 == 10:
                    self.learn()
            all_model_path = self.model_path + "/all_model/"
            best_model_path = self.model_path + "/best_model/"
            if not os.path.exists(all_model_path):
                os.makedirs(all_model_path)
            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)
            torch.save(
                self.asset_policy,
                all_model_path + "asset_policy_num_epoch_{}.pth".format(i))
            torch.save(
                self.asset_policy_critic,
                all_model_path + "asset_critic_num_epoch_{}.pth".format(i))
            torch.save(
                self.market_policy,
                all_model_path + "market_policy_num_epoch_{}.pth".format(i))
            print("validating")
            s = self.valid_env_instance.reset()
            done = False
            rewards = 0
            while not done:
                old_state = s
                old_market_state = torch.from_numpy(
                    make_market_information(
                        self.valid_env_instance.data,
                        technical_indicator=self.technical_indicator)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_old = make_correlation_information(
                    self.valid_env_instance.data)
                weights = self.compute_weights_test(
                    s,
                    make_market_information(
                        self.valid_env_instance.data,
                        technical_indicator=self.technical_indicator),
                    corr_matrix_old)
                s, reward, done, _ = self.valid_env_instance.step(weights)
                rewards += reward
            rewards_list.append(rewards)
        index = rewards_list.index(np.max(rewards_list))
        asset_policy_model_path = all_model_path + "asset_policy_num_epoch_{}.pth".format(
            index)
        asset_critic_model_path = all_model_path + "asset_critic_num_epoch_{}.pth".format(
            index)
        market_policy_model_path = all_model_path + "market_policy_num_epoch_{}.pth".format(
            index)
        self.asset_policy = torch.load(asset_policy_model_path)
        self.asset_policy_critic = torch.load(asset_critic_model_path)
        self.market_policy = torch.load(market_policy_model_path)
        torch.save(self.asset_policy, best_model_path + "asset_policy.pth")
        torch.save(self.asset_policy_critic,
                   best_model_path + "asset_critic.pth")
        torch.save(self.market_policy, best_model_path + "market_policy.pth")

    def test(self):
        s = self.test_env_instance.reset()
        done = False
        while not done:
            corr_matrix_old = make_correlation_information(
                self.test_env_instance.data)
            weights = self.compute_weights_test(
                s,
                make_market_information(
                    self.test_env_instance.data,
                    technical_indicator=self.technical_indicator),
                corr_matrix_old)
            s, reward, done, _ = self.test_env_instance.step(weights)
        df_return = self.test_env_instance.save_portfolio_return_memory()
        df_assets = self.test_env_instance.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values

        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        print(df)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        df.to_csv(self.result_path + "/result.csv")


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.autograd.set_detect_anomaly(True):
        a = trader(args)
        a.train_with_valid()
        a.test()
