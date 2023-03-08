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
        self.terminal = False
        self.money_sold = 0
        self.private_state_list = [self.private_state] * self.state_length

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
        self.private_state_list = [self.private_state] * self.state_length
        return np.array(self.public_imperfect_state), {
            "perfect_state": np.array(self.public_perfect_state),
            "private_state": np.array([self.private_state_list])
        }

    def step(self, action):
        # based on the current price information, we decided whether to trade use the next day's price
        # the reward is calculated as at*(p_(t+1)-average(p))
        self.day = self.day + 1

        self.terminal = (self.day >= (len(self.df) - self.state_length))
        if self.terminal:
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            leftover_day, leftover_order = self.private_state
            self.data_public_imperfect = self.df.iloc[
                                         self.day - self.state_length:self.day, :]
            current_price = self.data_public_imperfect.iloc[-1].close
            self.money_sold += leftover_order * current_price
            self.public_imperfect_state = np.array(self.public_imperfect_state)
            self.private_state_list.append([0, 0])
            self.private_state_list.remove(self.private_state_list[0])

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
                "money_sold":self.money_sold
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
            if np.abs(action) < np.abs(leftover_order):
                self.money_sold += action * current_price
                self.reward = action * (
                        current_price / previous_average_price - 1)
            else:
                self.money_sold += leftover_order * current_price
                self.reward = leftover_order * (
                        current_price / previous_average_price - 1)
                self.terminal = True
                stats = OrderedDict(
                {
                    "Money Sold": ["{:04f}".format(self.money_sold)],
                }
                )
                table = print_metrics(stats)
                print(table)
            leftover_day, leftover_order = leftover_day - 1 / (len(
                self.df) - 2 * self.state_length), leftover_order - action
            self.private_state = np.array([leftover_day, leftover_order])
            self.private_state_list.append(self.private_state)
            self.private_state_list.remove(self.private_state_list[0])
            return self.public_imperfect_state, self.reward, self.terminal, {
                "perfect_state": np.array([self.public_perfect_state]),
                "private_state": np.array([self.private_state_list]),
                "money_sold": self.money_sold
            }

    def find_money_sold(self):
        return self.money_sold