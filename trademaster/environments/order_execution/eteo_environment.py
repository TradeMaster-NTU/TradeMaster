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
class OrderExecutionETEOEnvironment(Environments):
    def __init__(self, **kwargs):
        super(OrderExecutionETEOEnvironment, self).__init__()

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
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])
        self.target_order = get_attr(self.dataset, "target_order", 1)
        self.portfolio = [self.initial_amount] + [0] + [0] + [0]
        self.portfolio_value_history = [self.initial_amount]
        self.portfolio_history = [self.portfolio]
        self.order_length = get_attr(kwargs, "length_keeping", 30)

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
            self.start_date=self.df.loc[:,'date'].iloc[0]
            self.end_date = self.df.loc[:,'date'].iloc[-1]
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)



        self.time_frame = 0
        self.order_history = []
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        # set action第一个为volume 正为买 负为卖 第二个为价格
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list) + 2,))

        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        self.data = self.df.loc[self.time_frame, :]
        self.data_normal = self.data.copy()
        self.data_normal["order_money"] = self.data_normal[
                                              "buys"] + self.data_normal["sells"]
        self.data_normal["buys"] = self.data_normal["buys"] / self.data_normal[
            "order_money"]
        self.data_normal["sells"] = self.data_normal[
                                        "sells"] / self.data_normal["order_money"]
        self.data_normal["midpoint"] = self.data_normal[
                                           "midpoint"] / self.data_normal["order_money"]

        max_value = max(self.data_normal[self.tech_indicator_list])
        self.data_normal[self.tech_indicator_list] = self.data_normal[
                                                         self.tech_indicator_list] / max_value
        self.public_state = self.data_normal[
            self.tech_indicator_list].values.tolist()
        self.terminal = False
        self.rewards = 0

        data_left = [(len(self.df.index.unique()) - self.time_frame) / 3600]
        order_left = [self.target_order]
        self.private_state = data_left + order_left
        self.state = np.array(self.public_state + self.private_state)

    def reset(self):
        self.time_frame = 0
        self.portfolio = [self.initial_amount] + [0] + [0] + [0]
        self.portfolio_history = [self.portfolio]
        self.order_history = []
        self.data = self.df.loc[self.time_frame, :]
        self.data_normal = self.data.copy()
        self.data_normal["order_money"] = self.data_normal[
                                              "buys"] + self.data_normal["sells"]
        self.data_normal["buys"] = self.data_normal["buys"] / self.data_normal[
            "order_money"]
        self.data_normal["sells"] = self.data_normal[
                                        "sells"] / self.data_normal["order_money"]
        self.data_normal["midpoint"] = self.data_normal[
                                           "midpoint"] / self.data_normal["order_money"]
        max_value = max(self.data_normal[self.tech_indicator_list])
        self.data_normal[self.tech_indicator_list] = self.data_normal[
                                                         self.tech_indicator_list] / max_value
        self.public_state = self.data_normal[
            self.tech_indicator_list].values.tolist()
        self.terminal = False
        self.rewards = 0
        data_left = [(len(self.df.index.unique()) - self.time_frame) / 3600]
        order_left = [self.target_order]
        self.private_state = data_left + order_left
        self.state = np.array(self.public_state + self.private_state)
        return self.state

    def step(self, action: np.array):

        # for now the target and step is a little different from the origional one:
        # there are 2 modification, first, we need bear the left order and left time in our head and at the end of the day, we need to
        # finish all the left order.
        # for the step ones we first check the dimension, then we check whether it is terminal
        # if it is not terminal, we first check whether you want to buy or sell
        # if you want to buy, we first use the single price times the amount and compare it with our cash, if you do not have enough cash, we will
        # use the all the cash divded by the single price you ask and make an order like this(TODO: this is a little bizzare because we will check the order again
        # when we have a chance to trade and it is largely depend on the cash back then if your cash is not enough we just place the order we can
        # afford and cancell the rest )
        # if you want to sell, we first have to check the whether you are currently holding and if you do not have enough bitcoin, then we will shrink it to the share
        # you have at this moment and hold the order, we will double check it at the moment when trades could happen and if we do not have enough share
        # we will cancell the rest order as well

        # check the dimension
        if action.shape != (2,):
            raiseExceptions(
                "sorry, the dimension of action is not correct, the dimension should be (2,), where the first dimension is the \
                volume you want to trade and the second dimension is the price you wanna sell or buy"
            )
        # check the amount of money you have or bitcoin is enough for the current order
        self.portfolio = self.portfolio_history[-1]
        # portfolio 有4个元素 1. free cash 2. cash will be used for existing order in the LOB 3.free bitcoin 4. bitcoin in order that is listed in the order book for sell
        # notice 1+2 the amount of cash we are currntely holding and 3+4 is the bitcoin we are holding
        # according to the order 1 and 3 we modify our action and make the order tradable
        # TODO we need to modify the definition of portfolio and the very end as well and add it into the portfolio_history part
        # the think is that we check the order history and clarify the cash and bitcoin taken in the order book and
        action[1]=action[1]
        if action[0] < 0:
            # if the action's volume is smaller than 0, we are going to sell the bitcoin we are holding
            sell_volume = -action[0]
            if self.portfolio[2] <= sell_volume:
                sell_volume = self.portfolio[2]
            action = [-sell_volume, action[1]]
            if action[1] < self.data["midpoint"] * (
                    1 + self.data["asks_distance_0"] *
                    0.01) or action[1] > self.data["midpoint"] * (
                    1 + self.data["asks_distance_14"] * 0.01):
                start_volume = 0
            else:
                start_volume = 0
                for i in range(15):
                    if action[1] == self.data["midpoint"] * (
                            1 +
                            self.data["asks_distance_{}".format(i)] * 0.01):
                        start_volume = self.data["asks_notional_{}".format(i)]
            action = [-sell_volume, action[1], start_volume]

        if action[0] > 0:
            # if the action's volume is greater than 0, we are going to buy the bitcoin we are holding
            buy_volume = action[0]
            buy_money = buy_volume * action[1]
            if buy_money >= self.portfolio[0]:
                buy_volume = self.portfolio[0] / action[1]
            action = [buy_volume, action[1]]
            if action[1] > self.data["midpoint"] * (
                    1 + self.data["bids_distance_0"] *
                    0.01) or action[1] < self.data["midpoint"] * (
                    1 + self.data["bids_distance_14"] * 0.01):
                start_volume = 0
            else:
                start_volume = 0
                for i in range(15):
                    if action[1] == self.data["midpoint"] * (
                            1 + self.data["bids_distance_{}".format(i)]):
                        start_volume = self.data["bids_notional_{}".format(i)]
            action = [buy_volume, action[1], start_volume]
        if action[0] == 0:
            action = [0, 0, 0]
        order = action
        self.order_history.append(order)
        if len(self.order_history) > self.order_length:
            self.order_history.pop(0)
        # 目前我的order_history中全部的订单已经修改完毕 现在开始计算里面是否有符合的订单选项
        # 因为我们是向后插入 因此越早前面的是我们越早进行挂单的order 从前向后检查
        # 检查的具体流程 首先我们step到下一个时间点 然后修正start volume：我们先把start volume不为0的order挑出来 然后找到目前时间段对应的level 若找不到直接
        # 归0 若找得到则利用cancel_order调整数量 直到start_volume为0开始
        previous_data = self.data
        self.time_frame = self.time_frame + 1
        self.data = self.df.loc[self.time_frame, :]
        for i in range(len(self.order_history)):
            order = self.order_history[i]
            start_volume = order[2]
            if order[0] < 0:
                # 要卖 看ask
                if order[2] != 0:
                    if order[1] < self.data["midpoint"] * (
                            1 + self.data["asks_distance_0"] *
                            0.01) or order[1] > self.data["midpoint"] * (
                            1 + self.data["asks_distance_14"] * 0.01):
                        order[2] = 0
                    else:
                        order[2] = 0
                        for i in range(15):
                            if order[1] == self.data["midpoint"] * (
                                    1 + self.data["asks_distance_{}".format(i)]
                                    * 0.01):
                                order[2] = max(
                                    0, start_volume - self.data[
                                        "asks_cancel_notional_{}".format(i)])
            if order[0] > 0:
                if order[2] != 0:
                    if order[1] > self.data["midpoint"] * (
                            1 + self.data["bids_distance_0"] *
                            0.01) or order[1] < self.data["midpoint"] * (
                            1 + self.data["bids_distance_14"] * 0.01):
                        order[2] = 0
                    else:
                        order[2] = 0
                        for i in range(15):
                            if order[1] == self.data["midpoint"] * (
                                    1 + self.data["bids_distance_{}".format(i)]
                                    * 0.01):
                                order[2] = max(
                                    0, start_volume - self.data[
                                        "bids_cancel_notional_{}".format(i)])
            self.order_history[i] = order
        # 更新过start_volume之后进行从前到后的order能否被执行的审查 并一次来更新self.portfolio和self.portfolio_history
        # 先进行self.portfolio的更新（因为我们已经place an order 所以自由与在orderbook中的调配已经发生了变换 所以需要先调整一次）
        # 调整后根据调整的self.portfolio 在进行order是否能被执行的问题 然后再进行self.portfolio的调整 如果有变化
        all_cash = self.portfolio[0] + self.portfolio[1]
        all_bitcoin = self.portfolio[2] + self.portfolio[3]
        old_portfolio_value = self.portfolio[0] + self.portfolio[
            1] + previous_data["midpoint"] * (self.portfolio[2] +
                                              self.portfolio[3])
        self.portfolio_value_history.append(old_portfolio_value)
        ordered_cash = 0
        ordered_bitcoin = 0
        for i in range(len(self.order_history)):
            if self.order_history[i][0] < 0:
                ordered_bitcoin = ordered_bitcoin - self.order_history[i][0]
            if self.order_history[i][0] > 0:
                ordered_cash = ordered_cash + self.order_history[i][
                    0] * self.order_history[i][1]
        free_cash = all_cash - ordered_cash
        free_bitcoin = all_bitcoin - ordered_bitcoin
        if np.abs(free_cash) < 1e-10:
            free_cash = 0
        self.portfolio = [
            free_cash, ordered_cash, free_bitcoin, ordered_bitcoin
        ]
        if free_cash < 0 or free_bitcoin < 0:
            raise Exception(
                "Something is wrong witht the order you place and there is no enough free cash or bitcoin in our portfolios, \
            the current portfolio is {}".format(self.portfolio))
            # order execution 的portfolio转移： 从ordered_cash转向free bitcoin 或者从ordered bitcoin 转向free cash
        for i in range(len(self.order_history)):
            order = self.order_history[i]
            if order[0] < 0:
                # 要卖的执行得看买的bid的最好distance够不够 如果够就可以根绝volume执行 若不够 怎不能进行交易
                if order[1] < self.data["midpoint"] * (
                        1 + self.data["bids_distance_0"] * 0.01):
                    # the order could be executed now let us see what is the greatest bargin
                    # 能划过去 默认直接全都能吃下去
                    self.portfolio = [
                        self.portfolio[0] - order[0] * order[1],
                        self.portfolio[1], self.portfolio[2],
                        self.portfolio[3] + order[0]
                    ]
                    order = [0, 0, 0]
                elif order[1] == self.data["midpoint"] * (
                        1 + self.data["bids_distance_0"] * 0.01):
                    # 看volume了
                    tradable_volume = min(
                        self.data["bids_notional_0"] - order[2], -order[0])
                    if tradable_volume == -order[0]:
                        self.portfolio = [
                            self.portfolio[0] - order[0] * order[1],
                            self.portfolio[1], self.portfolio[2],
                            self.portfolio[3] + order[0]
                        ]
                        order = [0, 0, 0]
                    else:
                        order[0] = order[0] + tradable_volume
                        order[2] = 0
                        self.portfolio = [
                            self.portfolio[0] + tradable_volume * order[1],
                            self.portfolio[1], self.portfolio[2],
                            self.portfolio[3] - tradable_volume
                        ]
            if order[0] > 0:
                if order[1] > self.data["midpoint"] * (
                        1 + self.data["asks_distance_0"] * 0.01):
                    self.portfolio = [
                        self.portfolio[0],
                        self.portfolio[1] - order[0] * order[1],
                        self.portfolio[2] + order[0], self.portfolio[3]
                    ]
                    order = [0, 0, 0]
                elif order[1] == self.data["midpoint"] * (
                        1 + self.data["asks_distance_0"] * 0.01):
                    tradable_volume = min(
                        self.data["asks_notional_0"] - order[2], order[0])
                    if tradable_volume == order[0]:
                        self.portfolio = [
                            self.portfolio[0],
                            self.portfolio[1] - order[0] * order[1],
                            self.portfolio[2] + order[0], self.portfolio[3]
                        ]
                        order = [0, 0, 0]
                    else:
                        order[0] = order[0] - tradable_volume
                        order[2] = 0
                        self.portfolio = [
                            self.portfolio[0],
                            self.portfolio[1] - tradable_volume * order[1],
                            self.portfolio[2] + tradable_volume,
                            self.portfolio[3]
                        ]
            self.order_history[i] = order
        self.portfolio_history.append(self.portfolio)
        new_portfolio_value = self.portfolio[0] + self.portfolio[
            1] + self.data["midpoint"] * (self.portfolio[2] +
                                          self.portfolio[3])
        self.portfolio_value_history.append(new_portfolio_value)
        self.data_normal = self.data.copy()
        self.data_normal["order_money"] = self.data_normal[
                                              "buys"] + self.data_normal["sells"]
        self.data_normal["buys"] = self.data_normal["buys"] / self.data_normal[
            "order_money"]
        self.data_normal["sells"] = self.data_normal[
                                        "sells"] / self.data_normal["order_money"]
        self.data_normal["midpoint"] = self.data_normal[
                                           "midpoint"] / self.data_normal["order_money"]
        max_value = max(self.data_normal[self.tech_indicator_list])
        self.data_normal[self.tech_indicator_list] = self.data_normal[
                                                         self.tech_indicator_list] / max_value
        self.public_state = self.data_normal[
            self.tech_indicator_list].values.tolist()
        left_order = self.target_order - (self.portfolio[2] +
                                          self.portfolio[3])

        left_date = [(len(self.df.index.unique()) - self.time_frame) / 3600]
        self.private_state = left_date + [left_order]
        self.state = np.array(self.public_state + self.private_state)
        self.terminal = (self.time_frame + 1 >= self.df.index[-1])
        if self.terminal:
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            # 终结时候计算reward
            # 先进行IS TWAP的计算
            # becasue the bitcoin at the end is the same and therefore we use the
            all_length = len(self.df.index)
            single_volume = self.target_order / all_length
            cash_used = 0
            for index in self.df.index:
                cash_used = cash_used + single_volume * self.df.iloc[index][
                    "midpoint"] * (1 + self.data["asks_distance_0"] * 0.01)
            TWAP_value = self.initial_amount - cash_used
            cash_left = self.portfolio[0] + self.portfolio[1]
            if left_order > 0:
                # we need to buy more but we do not have the time, so we have to buy it at the lowest price we can get
                cash_left = cash_left - left_order * self.data["midpoint"] * (
                        1 + self.data["asks_distance_0"] * 0.01)
            if left_order < 0:
                cash_left = cash_left - left_order * self.data["midpoint"] * (
                        1 + self.data["bids_distance_0"] * 0.01)
            if cash_left >= TWAP_value:
                self.reward = 1
            else:
                self.reward = 0

            stats = OrderedDict(
                {
                    "Cash Left": ["{:04f}".format(cash_left)],
                    "TWAP": ["{:04f}".format(TWAP_value)],
                    "Cash Left Ratio": ["{:04f}%".format(100 * (cash_left - TWAP_value) / TWAP_value)],
                }
            )
            table = print_metrics(stats)
            print(table)
            self.cash_left=cash_left

            return self.state, self.reward, self.terminal, {'cash_left':cash_left,'TWAP_value':TWAP_value}
        else:
            return self.state, 0, self.terminal, {}