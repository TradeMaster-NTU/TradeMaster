from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import numpy as np
from trademaster.utils import get_attr, print_metrics
import pandas as pd
from ..custom import Environments
from ..builder import ENVIRONMENTS
from gym import spaces
from collections import OrderedDict


@ENVIRONMENTS.register_module()
class AlgorithmicTradingEnvironment(Environments):
    def __init__(self, **kwargs):
        super(AlgorithmicTradingEnvironment, self).__init__()

        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")

        self.df_path = None
        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)

        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])
        self.forward_num_day = get_attr(self.dataset, "forward_num_day", [])
        self.backward_num_day = get_attr(self.dataset, "backward_num_day", [])
        self.max_volume = get_attr(self.dataset, "max_volume", 1)
        self.future_weights = get_attr(self.dataset, "future_weights", 0.2)

        if self.task.startswith("test_style"):
            style_test_path = get_attr(kwargs, "style_test_path", None)
            self.df = pd.read_csv(style_test_path, index_col=0)
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)

        self.action_space = spaces.Discrete(2 * (self.max_volume) + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list) * self.backward_num_day +
                   2,),
        )

        self.action_dim = self.action_space.n
        self.state_dim = self.observation_space.shape[0]

        # reset
        self.compound_memory = [[self.initial_amount, 0]]
        # the compound_memory's element consists of 2 parts: the cash and the number of bitcoin you have in hand
        self.portfolio_return_memory = [0]
        self.transaction_cost_memory = []
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.day = self.backward_num_day
        self.data = self.df.iloc[self.day - self.backward_num_day:self.day, :]
        self.date_memory = [self.data.date.unique()[-1]]
        self.state = [
            self.data[tech].values.tolist()
            for tech in self.tech_indicator_list
        ]
        self.state = np.array(self.state).reshape(-1).tolist()
        self.state = self.state + self.compound_memory[-1]
        self.state = np.array(self.state)

    def reset(self):
        # here is a little difference: we only have one asset
        # it starts with the back_num_day and ends in end-self.forward_num_day
        # for the information, it should calculate 2 additional things
        self.compound_memory = [[self.initial_amount, 0]]
        self.portfolio_return_memory = [0]
        self.transaction_cost_memory = []
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.day = self.backward_num_day
        self.data = self.df.iloc[self.day - self.backward_num_day:self.day, :]
        self.date_memory = [self.data.date.unique()[-1]]
        self.state = [
            self.data[tech].values.tolist()
            for tech in self.tech_indicator_list
        ]
        self.state = np.array(self.state).reshape(-1).tolist()
        self.state = self.state + self.compound_memory[-1]
        self.state = np.array(self.state)

        return self.state

    def step(self, action):
        # 此处的action应为仓位变化
        self.terminal = self.day >= len(
            self.df.index.unique()) - self.forward_num_day - 1
        if self.terminal:
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            stats = OrderedDict(
                {
                    "Profit Margin": ["{:04f}%".format(tr * 100)],
                    "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                    "Volatility": ["{:04f}".format(vol)],
                    "Max Drawdown": ["{:04f}".format(mdd)],
                    "Calmar Ratio": ["{:04f}".format(cr)],
                    "Sortino Ratio": ["{:04f}".format(sor)],
                }
            )
            table = print_metrics(stats)
            print(table)
            return self.state, self.reward, self.terminal, {
                "volidality": self.var
            }
        else:
            buy_volume = action - self.max_volume
            hold_volume = self.compound_memory[-1][1] + buy_volume
            cash_variation_number = np.abs(hold_volume) - np.abs(
                self.compound_memory[-1][1])
            if cash_variation_number < 0:
                # 我们卖出了一些比特币 并获得了一些额外的现金 没有支付不起手续费的情况
                cash = self.compound_memory[-1][0] + np.abs(
                    cash_variation_number) * self.data.iloc[-1, :].close * (
                               1 - self.transaction_cost_pct)
                hold_volume = hold_volume
            else:
                # 如果我们要放更多的钱到比特币手里 无论是买空还是卖空, 现金将会减少 会出现买不起的情况
                if self.compound_memory[-1][0] > np.abs(
                        buy_volume) * self.data.iloc[-1, :].close / (
                        1 - self.transaction_cost_pct):
                    # 如果手里的现金足够我们支付买卖比特币的费用外加手续费
                    cash = self.compound_memory[-1][0] - np.abs(
                        buy_volume) * self.data.iloc[-1, :].close / (
                                   1 - self.transaction_cost_pct)
                    hold_volume = hold_volume
                else:
                    # 如果没有足够的现金支持
                    max_trading = int(self.compound_memory[-1][0] /
                                      (self.data.iloc[-1, :].close /
                                       (1 - self.transaction_cost_pct)))
                    buy_volume = (np.abs(buy_volume) /
                                  buy_volume) * max_trading
                    hold_volume = self.compound_memory[-1][1] + buy_volume
                    cash = self.compound_memory[-1][0] - np.abs(
                        buy_volume) * self.data.iloc[-1, :].close / (
                                   1 - self.transaction_cost_pct)
            compound = [cash, hold_volume]
            self.compound_memory.append(compound)
            old_price = self.data.iloc[-1, :].close
            self.day = self.day + 1
            # 接下来 我们先进行reward的计算 而后进行各个部分历史的填充以及state的迭代
            self.data = self.df.iloc[self.day -
                                     self.backward_num_day:self.day, :]
            new_price = self.data.iloc[-1, :].close
            # -2来源：-1来自于df.iloc从0开始 -1来自于已经是新的self.day了 加过1了
            newer_price = self.df.iloc[self.day + self.forward_num_day -
                                       2].close
            # 计算reward 经过一天 价格发生变化 并提却更遥远的未来做估值打算
            self.reward = compound[1] * (
                    (new_price - old_price) + self.future_weights *
                    (newer_price - old_price))
            self.state = [
                self.data[tech].values.tolist()
                for tech in self.tech_indicator_list
            ]
            self.state = np.array(self.state).reshape(-1).tolist()
            self.state = self.state + self.compound_memory[-1]
            self.state = np.array(self.state)
            self.portfolio_return_memory.append(compound[1] *
                                                (new_price - old_price))
            self.portfolio_value = compound[0] + compound[1] * (new_price)
            self.asset_memory.append(self.portfolio_value)
            self.future_data = self.df.iloc[self.day - 1:self.day +
                                                         self.forward_num_day, :]
            self.date_memory.append(self.data.date.unique()[-1])
            close_price_list = self.future_data.close.tolist()
            labels = []
            for i in range(len(close_price_list) - 1):
                new_price = close_price_list[i + 1]
                old_price = close_price_list[i]
                return_rate = new_price / old_price - 1
                labels.append(return_rate)
            self.var = np.var(labels)

            return self.state, self.reward, self.terminal, {
                "volidality": self.var
            }

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
        daily_return = df["daily_return"]
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1] / (df["total assets"].values[0] + 1e-10) - 1
        sharpe_ratio = np.mean(daily_return) / (np.std(daily_return) * (len(df) ** 0.5) + 1e-10)
        vol = np.std(daily_return)
        mdd = max((max(df["total assets"]) - df["total assets"]) / (max(df["total assets"])) + 1e-10)
        cr = np.sum(daily_return) / (mdd + 1e-10)
        sor = np.sum(daily_return) / (np.std(neg_ret_lst) + 1e-10) / (np.sqrt(len(daily_return))+1e-10)
        return tr, sharpe_ratio, vol, mdd, cr, sor
