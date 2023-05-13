from __future__ import annotations

import sys
from logging import raiseExceptions
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
class OrderExecutionPDEnvironment(Environments):
    def __init__(self, **kwargs):
        super(OrderExecutionPDEnvironment, self).__init__()

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
        self.state_length = get_attr(self.dataset, "state_length", 10)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])
        self.target_order = get_attr(self.dataset, "target_order", 1)
        self.portfolio = [self.initial_amount] + [0] + [0] + [0]
        self.portfolio_value_history = [self.initial_amount]
        self.portfolio_history = [self.portfolio]
        self.order_length = get_attr(kwargs, "length_keeping", None)

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
            self.start_date=self.df.loc[:,'date'].iloc[0]
            self.end_date = self.df.loc[:,'date'].iloc[-1]
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)



        self.time_frame = 0
        self.order_history = []
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1, ),
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_length, len(self.tech_indicator_list)),
        )

        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[-1]
        self.public_state_dim = self.state_dim
        self.private_state_dim = 2

        self.day = self.state_length
        self.data_public_imperfect = self.df.iloc[
            self.day - self.state_length:self.day, :]
        self.data_public_perfect = self.df.iloc[
            self.day - self.state_length:self.day +
            self.state_length, :]
        self.public_imperfect_state = [
            self.data_public_imperfect[
                self.tech_indicator_list].values.tolist()
        ]
        self.public_perfect_state = [
            self.data_public_perfect[self.tech_indicator_list].values.tolist()
        ]
        # private state indicates the left date and order
        self.private_state = np.array([1, 1])
        self.private_state_start= self.private_state
        self.terminal = False
        self.money_sold = 0
        self.private_state_list = [self.private_state] * self.state_length
        self.money_sold_list = []
        self.action_list = []
        self.price_list = []
        self.asset_list = []

    def reset(self):
        self.terminal = False
        self.day = self.state_length
        self.data_public_imperfect = self.df.iloc[
                                     self.day - self.state_length:self.day, :]
        self.data_public_perfect = self.df.iloc[
                                   self.day - self.state_length:self.day +
                                                                    self.state_length, :]
        self.public_imperfect_state = [
            self.data_public_imperfect[
                self.tech_indicator_list].values.tolist()
        ]
        self.public_perfect_state = [
            self.data_public_perfect[self.tech_indicator_list].values.tolist()
        ]
        # private state indicates the
        self.private_state = np.array([1, 1])
        self.money_sold = 0
        self.money_sold_list = []
        self.action_list=[]
        self.private_state_list = [self.private_state] * self.state_length
        self.asset_list = []
        self.price_list = []
        return np.array(self.public_imperfect_state), {
            "perfect_state": np.array(self.public_perfect_state),
            "private_state": np.array([self.private_state_list])
        }

    def step(self, action):
        # based on the current price information, we decided whether to trade use the next day's price
        # the reward is calculated as at*(p_(t+1)-average(p))
        self.day = self.day + 1
        self.action_list.append(action)

        self.terminal = (self.day >= (len(self.df) - self.state_length))
        if self.terminal:
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            leftover_day, leftover_order = self.private_state
            self.data_public_imperfect = self.df.iloc[
                                         self.day - self.state_length:self.day, :]
            current_price = self.data_public_imperfect.iloc[-1].close
            self.price_list.append(current_price)
            self.money_sold += leftover_order * current_price
            self.money_sold_list.append(self.money_sold)
            self.public_imperfect_state = np.array(self.public_imperfect_state)
            self.private_state_list.append([0, 0])
            self.private_state_list.remove(self.private_state_list[0])

            self.asset_list.append(self.money_sold + leftover_order * current_price)
            stats = OrderedDict(
                {
                    "Money Sold": ["{:04f}".format(self.money_sold)],
                }
            )
            table = print_metrics(stats)
            print(table)

            return self.public_imperfect_state, self.reward, self.terminal, {
                "perfect_state": np.array([self.public_perfect_state]),
                "private_state": np.array([self.private_state_list]),
                "money_sold":self.money_sold,
                'money_sold_list':self.money_sold_list,
                'Total Asset':self.asset_list
            }

        else:
            leftover_day, leftover_order = self.private_state

            previous_average_price = np.mean(self.df.iloc[:self.day -
                                                           1].close.values)
            self.data_public_imperfect = self.df.iloc[
                                         self.day - self.state_length:self.day, :]
            self.data_public_perfect = self.df.iloc[
                                       self.day - self.state_length:self.day +
                                                                        self.state_length, :]
            self.public_imperfect_state = self.data_public_imperfect[
                self.tech_indicator_list].values.tolist()

            self.public_perfect_state = self.data_public_perfect[
                self.tech_indicator_list].values.tolist()

            self.public_imperfect_state = np.array(
                [self.public_imperfect_state])
            current_price = self.data_public_imperfect.iloc[-1].close
            self.price_list.append(current_price)
            if np.abs(action) < np.abs(leftover_order):
                self.money_sold += action * current_price
                self.reward = action * (
                        current_price / previous_average_price - 1)
            else:
                self.money_sold += leftover_order * current_price
                self.reward = leftover_order * (
                        current_price / previous_average_price - 1)
                self.terminal = True
                # calculate avg price in the self.price_list
                avg_money_sold = np.mean(self.price_list) * self.private_state_start[1]
                if self.task.startswith("test"):
                    stats = OrderedDict(
                    {
                        "Money Sold": ["{:04f}".format(self.money_sold)],
                        "Money Sold on average price in trading period": ["{:04f}".format(avg_money_sold)]
                    }
                    )
                else:
                    stats = OrderedDict(
                        {
                            "Money Sold": ["{:04f}".format(self.money_sold)],
                        }
                )
                table = print_metrics(stats)
                print(table)
            leftover_day, leftover_order = leftover_day - 1 / (len(
                self.df) - 2 * self.state_length), leftover_order - action
            self.money_sold_list.append(self.money_sold)
            self.private_state = np.array([leftover_day, leftover_order])
            self.private_state_list.append(self.private_state)
            self.private_state_list.remove(self.private_state_list[0])


            market_features_dict = {'close':self.df['close'].values.tolist()}
            buy_points={}
            sell_points={}
            for i,action in enumerate(self.action_list):
                # if the action's volume is greater than 0, we are going to buy the bitcoin we are holding
                if action > 0:
                    sell_points[i] = action
                elif action < 0:
                    buy_points[i] = action
            trading_points = {'buy':buy_points,'sell':sell_points}
            self.asset_list.append(self.money_sold + leftover_order * current_price)




            return self.public_imperfect_state, self.reward, self.terminal, {
                "perfect_state": np.array([self.public_perfect_state]),
                "private_state": np.array([self.private_state_list]),
                "money_sold": self.money_sold,
                'money_sold_list': self.money_sold_list,
                'Total Asset': self.asset_list,
                'trading_points':trading_points,
                'market_features_dict': market_features_dict
            }

    def find_money_sold(self):
        return self.money_sold