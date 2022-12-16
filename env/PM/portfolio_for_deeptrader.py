import configparser
from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--df_path",
    type=str,
    default="./data/data/dj30/test.csv",
    help="the path for the downloaded data to generate the environment")
parser.add_argument("--initial_amount",
                    type=int,
                    default=100000,
                    help="the initial amount of money for trading")
parser.add_argument("--transaction_cost_pct",
                    type=float,
                    default=0.001,
                    help="the transcation cost for us to ")
parser.add_argument("--tech_indicator_list",
                    type=list,
                    default=[
                        "high", "low", "open", "close", "adjcp", "zopen",
                        "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10",
                        "zd_15", "zd_20", "zd_25", "zd_30"
                    ],
                    help="the name of the features to predict the label")
parser.add_argument("--length_day",
                    type=int,
                    default=10,
                    help="the length of the day our state contains")
args = parser.parse_args()


# here is a modification of the origional version
# there we must set a upper bound for the amount of stock we can short otherwise is off limit
# so instead of use the step 3) here we use the a different cerinao, we simply doing long and short at the same time for different stock
# the action space is contious so basically it is just an add-up with a short position
# since we do not have any market unit scoring data, we decide to use only the asset score part and therefore there is no part as risk-free process
class Tradingenv(gym.Env):
    def __init__(self, config):
        self.day = config["length_day"] - 1
        self.df = pd.read_csv(config["df_path"], index_col=0)
        self.stock_dim = len(self.df.tic.unique())
        self.initial_amount = config["initial_amount"]
        self.transaction_cost_pct = config["transaction_cost_pct"]
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim
        self.length_day = config["length_day"]

        self.tech_indicator_list = config["tech_indicator_list"]

        self.action_space = spaces.Box(low=-5,
                                       high=5,
                                       shape=(self.action_space_shape, ))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list), self.state_space_shape,
                   self.length_day))

        self.data = self.df.loc[self.day - self.length_day + 1:self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        self.state = np.array([[
            self.data[self.data.tic == tic][tech].values.tolist()
            for tech in self.tech_indicator_list
        ] for tic in self.data.tic.unique()])
        # self.state = np.transpose(self.state, (2, 0, 2))
        # 此时返回的维度：(时间长度，tic数量，特征数量)
        #TCN目前只能处理单个时间序列 所以我们的想法是把tic数量当作batch_size 以特征数量当作channel数进行处理 最后返回符合Kl*tic数*特征数
        #deeptrader貌似并没有做fcl st tcn输入与输出的维度可以不同

        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []

    def reset(self):
        self.day = self.length_day - 1
        self.data = self.df.loc[self.day - self.length_day + 1:self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        self.state = np.array([[
            self.data[self.data.tic == tic][tech].values.tolist()
            for tech in self.tech_indicator_list
        ] for tic in self.data.tic.unique()])
        # self.state = np.transpose(self.state, (2, 0, 1))
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / (self.stock_dim)] * (self.stock_dim)]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []

        return self.state

    def step(self, weights):
        # make judgement about whether our data is running out
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        weights = np.array(weights)

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
            return self.state, 0, self.terminal, {"sharpe_ratio": sharpe_ratio}

        else:  #directly use the process of
            self.weights_memory.append(weights)
            last_day_memory = self.df.loc[self.day, :]
            self.day += 1
            self.data = self.df.loc[self.day - self.length_day + 1:self.day, :]
            self.state = np.array([[
                self.data[self.data.tic == tic][tech].values.tolist()
                for tech in self.tech_indicator_list
            ] for tic in self.data.tic.unique()])
            # self.state = np.transpose(self.state, (2, 0, 1))
            new_price_memory = self.df.loc[self.day, :]
            portfolio_weights = weights
            portfolio_return = sum(
                ((new_price_memory.close.values / last_day_memory.close.values)
                 - 1) * portfolio_weights)
            weights_brandnew = self.normalization(
                list(
                    np.array(weights[:]) * np.array(
                        (new_price_memory.close.values /
                         last_day_memory.close.values))))
            self.weights_memory.append(weights_brandnew)
            weights_old = (self.weights_memory[-3])
            weights_new = (self.weights_memory[-2])
            diff_weights = np.sum(
                np.abs(np.array(weights_old) - np.array(weights_new)))
            transcationfee = diff_weights * self.transaction_cost_pct * self.portfolio_value
            new_portfolio_value = (self.portfolio_value -
                                   transcationfee) * (1 + portfolio_return)
            portfolio_return = (new_portfolio_value -
                                self.portfolio_value) / self.portfolio_value
            self.reward = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[-1])
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.reward

        return self.state, self.reward, self.terminal, {}

    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        actions = np.array(actions)
        sum = np.abs(np.sum(actions))
        actions = actions / sum
        return actions

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


if __name__ == "__main__":
    # import yaml

    # def save_dict_to_yaml(dict_value: dict, save_path: str):
    #     with open(save_path, 'w') as file:
    #         file.write(yaml.dump(dict_value, allow_unicode=True))

    # def read_yaml_to_dict(yaml_path: str, ):
    #     with open(yaml_path) as file:
    #         dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
    #         return dict_value

    # save_dict_to_yaml(
    #     vars(args),
    #     "config/input_config/env/portfolio/portfolio_for_deeptrader/test.yml")

    df = pd.read_csv(vars(args)["df_path"])
    num_tickers = len(df.tic.unique())
    weights = [1 / num_tickers] * num_tickers
    print(num_tickers)
    a = Tradingenv(vars(args))
    state = a.reset()
    print(state.shape)
    done = False
    while not done:
        weights = [1 / num_tickers] * num_tickers
        state, reward, done, _ = a.step(weights)
        # print(state.shape)
        # print(reward)
    # print(a.save_portfolio_return_memory())
    # print(a.save_asset_memory())
