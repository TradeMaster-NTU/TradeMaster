import configparser
from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import sys

sys.path.append('./')

from agent.Investor_Imitator.model.net import MLP_reg
import torch.nn as nn
import torch.nn.init as init
import torch
import os
"""
env 需要重构的东西
state好说 主要是action
state的dimension num_tic*(tech_indicator_list(原来的state)+rank_name(MDD,AR,ER,WR,SR)(咱现在用logical_indicator 预测出来的))+num_tic(上一次的交易[0.1,0.2,...])
action 为discrete的logic_discriptor number. 现在重新构建step中的迭代过程：
有一个函数存portfolio的动作 有一个函数  有一个用网络预测结果的动作 要把之前的df换为tensor
action 选logic_discriptor, 用if映射一个1到5的一个logic_discriptor的选择
选择后用前面定义的evaluate函数得到预测的rank 然后softmax即可
config["tech_indicator_list"]为["high","low","open","close","adjcp",
            "zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10",
            "zd_15", "zd_20", "zd_25", "zd_30"
        ]
"""


class TradingEnv(gym.Env):
    def __init__(self, config):
        self.day = 0
        self.df = pd.read_csv(config["df_dict"], index_col=0)
        # determine how many network we need to import
        self.stock_dim = len(self.df.tic.unique())
        self.tic_list = self.df.tic.unique()
        """
 determine the overall logical_discriptor's dict to intergal in the later, the randomseed is the variable needed to be
        the same both in pre-trained logical discriptor and for the rl framework
        the dict in the example should be ./exeperinment/crypto/trained_model/12345
        to make the model actually work, we need all 5 logical indicator to work
        here we construct a double layer dict where the first layer's key is the type of the indicator and the second layer's key is the name of
        of the crypto, the contents of the second layer the torch load model we have trained.


        """

        self.network_dict = config["dict"]
        all_dict = {}
        for sub_file in os.listdir(self.network_dict):
            discriptor_path = os.path.join(self.network_dict, sub_file)
            best_model_path = "best_model"
            discriptor_best_path = os.path.join(discriptor_path,
                                                best_model_path)
            for net_dict in os.listdir(discriptor_best_path):

                indicator_dict = torch.load(
                    os.path.join(discriptor_best_path, net_dict)).cpu()

            all_dict.update({sub_file: indicator_dict})
        # here the self.net_2_dict is the 2 layer of dict and content is the network
        self.nets_2_dict = all_dict

        self.initial_amount = config["initial_amount"]
        self.transaction_cost_pct = config["transaction_cost_pct"]
        self.state_space_shape = self.stock_dim

        self.tech_indicator_list = config["tech_indicator_list"]

        self.action_space = spaces.Discrete(len(self.nets_2_dict))  #这动作从0开始

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.state_space_shape *
                   (len(self.nets_2_dict) + len(self.tech_indicator_list)) +
                   self.state_space_shape))

        self.data = self.df.loc[self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        # print(self.data.tic)
        tic_list = list(self.data.tic)
        tech_indicator_list = self.tech_indicator_list
        ARs = []
        SRs = []
        WRs = []
        MDDs = []
        ERs = []
        for i in range(len(tic_list)):
            tic_information = self.data[self.data.tic ==
                                        tic_list[i]][tech_indicator_list]
            tic_information = np.array(tic_information.values)
            tic_information = torch.from_numpy(tic_information)
            AR_model = all_dict["AR"]
            SR_model = all_dict["SR"]
            WR_model = all_dict["WR"]
            MDD_model = all_dict["MDD"]
            ER_model = all_dict["ER"]
            AR = AR_model(tic_information)
            AR = float(AR.detach().numpy())

            SR = SR_model(tic_information)
            SR = float(SR.detach().numpy())
            WR = WR_model(tic_information)
            WR = float(WR.detach().numpy())

            MDD = MDD_model(tic_information)
            MDD = float(MDD.detach().numpy())
            ER = ER_model(tic_information)
            ER = float(ER.detach().numpy())

            ARs.append(AR)
            SRs.append(SR)
            WRs.append(WR)
            MDDs.append(MDD)
            ERs.append(ER)

        st_pool = ARs + SRs + WRs + MDDs + ERs
        s_market = list(
            np.array(self.data[tech_indicator_list].values).reshape(-1))
        s_history_action = [1 / len(tic_list)] * len(tic_list)
        s = s_market + st_pool + s_history_action

        self.state = np.array(s)

        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]

        tic_list = list(self.data.tic)
        tech_indicator_list = self.tech_indicator_list
        ARs = []
        SRs = []
        WRs = []
        MDDs = []
        ERs = []
        for i in range(len(tic_list)):
            tic_information = self.data[self.data.tic ==
                                        tic_list[i]][tech_indicator_list]
            tic_information = np.array(tic_information.values)
            tic_information = torch.from_numpy(tic_information)
            AR_model = self.nets_2_dict["AR"]
            SR_model = self.nets_2_dict["SR"]
            WR_model = self.nets_2_dict["WR"]
            MDD_model = self.nets_2_dict["MDD"]
            ER_model = self.nets_2_dict["ER"]
            AR = AR_model(tic_information)
            AR = float(AR.detach().numpy())

            SR = SR_model(tic_information)
            SR = float(SR.detach().numpy())
            WR = WR_model(tic_information)
            WR = float(WR.detach().numpy())

            MDD = MDD_model(tic_information)
            MDD = float(MDD.detach().numpy())
            ER = ER_model(tic_information)
            ER = float(ER.detach().numpy())

            ARs.append(AR)
            SRs.append(SR)
            WRs.append(WR)
            MDDs.append(MDD)
            ERs.append(ER)

        st_pool = ARs + SRs + WRs + MDDs + ERs
        s_market = list(
            np.array(self.data[tech_indicator_list].values).reshape(-1))
        s_history_action = [1 / len(tic_list)] * len(tic_list)
        s = s_market + st_pool + s_history_action
        self.state = np.array(s)
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []
        return self.state

    def softmax(self, a):
        # b = a.copy()
        # b.sort()
        # ranks = []
        # for d in a:
        #     rank = b.index(d)
        #     ranks.append(rank)
        # dinominator = np.sum(np.exp(ranks))

        return np.exp(a) / np.sum(np.exp(a))

    def generate_portfolio_weights(self, actions):
        scores = []
        for i in range(len(self.tic_list)):
            tic_information = self.data[self.data.tic == self.tic_list[i]][
                self.tech_indicator_list]
            tic_information = np.array(tic_information.values)
            tic_information = torch.from_numpy(tic_information)
            AR_model = self.nets_2_dict["AR"]
            SR_model = self.nets_2_dict["SR"]
            WR_model = self.nets_2_dict["WR"]
            MDD_model = self.nets_2_dict["MDD"]
            ER_model = self.nets_2_dict["ER"]
            models = [AR_model, SR_model, WR_model, MDD_model, ER_model]
            actions = int(actions)
            if actions not in range(len(self.nets_2_dict)):
                raiseExceptions("the dimension is not correct")
            model = models[actions]
            score = model(tic_information)
            score = float(score.detach().numpy())
            scores.append(score)
        portfolio_weights = self.softmax(scores)
        return portfolio_weights

    def step(self, actions):
        #here the action change to from 0 to 4 which actually indicates a choice of logical discriptor and
        # make judgement about whether our data is running out
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            print("=================================")
            print("the profit margin is", tr * 100, "%")
            print("the sharpe ratio is", sharpe_ratio)
            print("the Volatility is", vol)
            print("the max drawdown is", mdd)
            print("the Calmar Ratio is", cr)
            print("the Sortino Ratio is", sor)
            print("=================================")
            return self.state, self.reward, self.terminal, {}

        else:
            # transfer actino into portofolios weights
            weights = self.generate_portfolio_weights(actions)
            self.weights_memory.append(weights)
            last_day_memory = self.data

            # step into the next time stamp
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # get the state
            ARs = []
            SRs = []
            WRs = []
            MDDs = []
            ERs = []
            for i in range(len(self.tic_list)):
                tic_information = self.data[self.data.tic == self.tic_list[i]][
                    self.tech_indicator_list]
                tic_information = np.array(tic_information.values)
                tic_information = torch.from_numpy(tic_information)
                AR_model = self.nets_2_dict["AR"]
                SR_model = self.nets_2_dict["SR"]
                WR_model = self.nets_2_dict["WR"]
                MDD_model = self.nets_2_dict["MDD"]
                ER_model = self.nets_2_dict["ER"]
                AR = AR_model(tic_information)
                AR = float(AR.detach().numpy())

                SR = SR_model(tic_information)
                SR = float(SR.detach().numpy())
                WR = WR_model(tic_information)
                WR = float(WR.detach().numpy())

                MDD = MDD_model(tic_information)
                MDD = float(MDD.detach().numpy())
                ER = ER_model(tic_information)
                ER = float(ER.detach().numpy())

                ARs.append(AR)
                SRs.append(SR)
                WRs.append(WR)
                MDDs.append(MDD)
                ERs.append(ER)

            st_pool = ARs + SRs + WRs + MDDs + ERs
            s_market = list(
                np.array(
                    self.data[self.tech_indicator_list].values).reshape(-1))
            s_history_action = list(self.weights_memory[-1])
            s = s_market + st_pool + s_history_action
            self.state = np.array(s)

            # get the portfolio return and the new weights(after one day's price variation, the weights will be a little different from
            # the weights when the action is first posed)
            portfolio_weights = weights[:]
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) *
                portfolio_weights)
            weights_brandnew = self.normalization(
                list(
                    np.array(weights[:]) * np.array(
                        (self.data.close.values /
                         last_day_memory.close.values))))
            self.weights_memory.append(weights_brandnew)

            # caculate the transcation fee(there could exist an error of about 0.1% when calculating)
            weights_old = (self.weights_memory[-3])
            weights_new = (self.weights_memory[-2])

            diff_weights = float(
                np.sum(np.abs(np.array(weights_old) - np.array(weights_new))))
            # print(diff_weights)
            # print(self.transaction_cost_pct)
            # print(self.portfolio_value)

            transcationfee = diff_weights * self.transaction_cost_pct * self.portfolio_value
            # print(transcationfee)

            # calculate the overal result
            new_portfolio_value = (self.portfolio_value -
                                   transcationfee) * (1 + portfolio_return)
            portfolio_return = (new_portfolio_value -
                                self.portfolio_value) / self.portfolio_value
            self.reward = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.reward

        return self.state, self.reward, self.terminal, {}

    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        actions = np.array(actions)
        sum = np.sum(actions)
        actions = actions / sum
        return actions

    # def softmax(self, actions):
    #     numerator = np.exp(actions)
    #     denominator = np.sum(np.exp(actions))
    #     softmax_output = numerator / denominator
    #     return softmax_output

    def save_portfolio_return_memory(self):
        # a record of return for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = df_date.date

        return df_return

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = df_date.date

        return df_value

    def evaualte(self, df):
        # a function to analysis the return & risk using history record
        daily_return = df["daily_return"]
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1] / df["total assets"].values[0] - 1
        sharpe_ratio = np.mean(daily_return) / \
            np.std(daily_return)*(len(df)**0.5)
        vol = np.std(daily_return)
        mdd = max((max(df["total assets"]) - df["total assets"]) /
                  max(df["total assets"]))
        cr = np.sum(daily_return) / mdd
        sor = np.sum(daily_return)/np.std(neg_ret_lst) / \
            np.sqrt(len(daily_return))
        return tr, sharpe_ratio, vol, mdd, cr, sor

    def analysis_result(self):
        # A simpler API for the environment to analysis itself when coming to terminal
        df_return = self.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df_value = self.save_asset_memory()
        assets = df_value["total assets"].values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        return self.evaualte(df)
