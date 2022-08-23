import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd


class TradingEnv(gym.Env):
    """TradingEnv is a RL trading environment based on gym.

    To initilize it, it needs config, which contains df,initial_amount,transaction_cost_pct,tech_indicator_list
    config:
    df: the data we need to simulate the market, usually detrived from the history of the market
    df should be in the form of below(this is the sz50's data)

                        date   open   high    low  close  ...     zd_10     zd_15     zd_20     zd_25     zd_30
0     2016-06-01 09:30:00.000  18.30  18.30  18.30  18.30  ...  0.008047  0.005478  0.003774  0.003374  0.004391
0     2016-06-01 09:30:00.000  27.75  27.75  27.75  27.75  ...  0.004826  0.003153  0.003185  0.004324  0.005856
0     2016-06-01 09:30:00.000   9.12   9.12   9.12   9.12  ...  0.022997  0.028054  0.032395  0.029880  0.028914
0     2016-06-01 09:30:00.000   4.83   4.83   4.83   4.83  ...  0.002703  0.000480  0.000180  0.001153  0.002342
0     2016-06-01 09:30:00.000  16.79  16.79  16.79  16.79  ... -0.000248  0.000000  0.001365  0.001092  0.001241
...                       ...    ...    ...    ...    ...  ...       ...       ...       ...       ...       ...
3154  2018-12-28 15:00:00.000   5.63   5.70   5.61   5.70  ... -0.016140 -0.016140 -0.017105 -0.013263 -0.007544
3154  2018-12-28 15:00:00.000  16.18  16.23  16.10  16.20  ... -0.005679 -0.014074 -0.019167 -0.018840 -0.016605
3154  2018-12-28 15:00:00.000   7.23   7.23   7.20   7.21  ...  0.006657  0.006935  0.007004  0.010541  0.012714
3154  2018-12-28 15:00:00.000  60.00  60.44  59.86  60.20  ... -0.012641 -0.017320 -0.016894 -0.017209 -0.019109

the index should be in the same page of date and the columns must contain all the tech_indicator_list in the config
    initial_amount:the initial amount of cash we let the agent hold at the very begining
    transaction_cost_pct: the percentage of the cost of each transaction 
    tech_indicator_list: the observational attributes that we want the agents to see 


the state in this environment after one action is an array with size (1,num_stock*len(tech_indicator_list)),
meaning the day after your transcation's market information: the stock and their tech_indicator_list
action is a array with the size (1+num_stock), meaning the portion of cash and each assets in your current portfolios
action is taken when the market is about to close, and when the market opens, the value of your portfolios varies, making the portions of each
assets varies and reward is the increase of your whole portfolios' value when the market is about to close again minus the transcation fee.
the state comes to an end when our data runs out

this environment also helps store the history of the cost fees and the daily reward.
It can helps you get a dataframe or a graph of the cost fees and the daily reward along the date as well when reaches a terminal and a 
comprehensive analysis of our investment record


    """
    def __init__(self, config):
        self.day = 0
        self.df = pd.read_csv(config["df_dict"], index_col=0)
        self.stock_dim = len(self.df.tic.unique())
        self.initial_amount = config["initial_amount"]
        self.transaction_cost_pct = config["transaction_cost_pct"]
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim + 1

        self.tech_indicator_list = config["tech_indicator_list"]

        self.action_space = spaces.Box(low=-5,
                                       high=5,
                                       shape=(self.action_space_shape, ))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list), self.state_space_shape))

        self.data = self.df.loc[self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        self.state = np.array([
            self.data[tech].values.tolist()
            for tech in self.tech_indicator_list
        ])

        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1] + [0] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.state = [
            self.data[tech].values.tolist()
            for tech in self.tech_indicator_list
        ]
        self.state = np.array(self.state)
        self.portfolio_value = self.initial_amount
        self.portfolio_return_memory = [0]

        self.terminal = False
        self.weights_memory = [[1] + [0] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []
        return self.state

    def step(self, actions):
        # make judgement about whether our data is running out
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        actions = np.array(actions)

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
            return self.state, self.reward, self.terminal, {
                "sharpe_ratio": sharpe_ratio
            }

        else:
            # transfer actino into portofolios weights
            weights = self.softmax(actions)
            self.weights_memory.append(weights)
            last_day_memory = self.data

            # step into the next time stamp
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # get the state
            self.state = np.array([
                self.data[tech].values.tolist()
                for tech in self.tech_indicator_list
            ])
            self.state = np.array(self.state)

            # get the portfolio return and the new weights(after one day's price variation, the weights will be a little different from
            # the weights when the action is first posed)
            portfolio_weights = weights[1:]
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) *
                portfolio_weights)
            weights_brandnew = self.normalization([weights[0]] + list(
                np.array(weights[1:]) * np.array(
                    (self.data.close.values / last_day_memory.close.values))))
            self.weights_memory.append(weights_brandnew)

            # caculate the transcation fee(there could exist an error of about 0.1% when calculating)
            weights_old = (self.weights_memory[-3])
            weights_new = (self.weights_memory[-2])

            diff_weights = np.sum(
                np.abs(np.array(weights_old) - np.array(weights_new)))
            transcationfee = diff_weights * self.transaction_cost_pct * self.portfolio_value

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

    def softmax(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

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
