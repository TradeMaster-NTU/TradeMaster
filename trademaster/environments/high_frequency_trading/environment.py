from __future__ import annotations
import torch
from collections import OrderedDict
from gym import spaces
from ..builder import ENVIRONMENTS
from ..custom import Environments
import pandas as pd
from trademaster.utils import get_attr, print_metrics
import numpy as np

import sys
from pathlib import Path
import pickle
import os.path as osp

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

__all__ = ["HighFrequencyTradingEnvironment", "HighFrequencyTradingTrainingEnvironment"]

@ENVIRONMENTS.register_module()
class HighFrequencyTradingEnvironment(Environments):
    def __init__(self, **kwargs):
        super(HighFrequencyTradingEnvironment, self).__init__()

        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")
        self.test_dynamic = int(get_attr(kwargs, "test_dynamic", "-1"))
        self.task_index = int(get_attr(kwargs, "task_index", "-1"))
        self.work_dir = get_attr(kwargs, "work_dir", "")

        self.df_path = None
        if self.task.startswith("train"):
            raise Exception(
                "the trading environment is only designed for testing or validing"
            )
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)

        self.transaction_cost_pct = get_attr(
            self.dataset, "transaction_cost_pct", 0.00005
        )
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])
        self.stack_length = get_attr(self.dataset, "backward_num_timestamp", 1)
        self.max_holding_number = get_attr(self.dataset, "max_holding_number", 0.01)

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
            self.start_date=self.df.loc[:,'date'].iloc[0]
            self.end_date = self.df.loc[:,'date'].iloc[-1]
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)


        self.action_space = spaces.Discrete(get_attr(self.dataset, "num_action", 11))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list) * self.stack_length,),
        )

        self.action_dim = self.action_space.n
        self.state_dim = self.observation_space.shape[0]
        self.demonstration_action = self.making_multi_level_dp_demonstration(
            max_punish=get_attr(self.dataset, "max_punish", 1e12)
        )
        # reset
        self.terminal = False
        self.day = self.stack_length
        self.data = self.df.iloc[self.day - self.stack_length : self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = 0
        self.position = 0
        self.reward_history = [0]
        self.test_id = 'agent'

    def sell_value(self, price_information, position):
        orgional_position = position
        # use bid price and size to evaluate
        value = 0
        # position 表示剩余的单量
        for i in range(1, 6):
            if position < price_information["bid{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["bid{}_size".format(i)]
                value += (
                    price_information["bid{}_price".format(i)]
                    * price_information["bid{}_size".format(i)]
                )
        if position > 0 and i == 5:
            # print("the holding could not be sell all clear")
            # 执行的单量
            actual_changed_position = orgional_position - position
        else:
            value += price_information["bid{}_price".format(i)] * position
            actual_changed_position = orgional_position
        # 卖的时候的手续费相当于少卖钱了
        self.comission_fee_history.append(self.transaction_cost_pct * value)

        return value * (1 - self.transaction_cost_pct), actual_changed_position

    def buy_value(self, price_information, position):
        # this value measure how much
        value = 0
        orgional_position = position
        for i in range(1, 6):
            if position < price_information["ask{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["ask{}_size".format(i)]
                value += (
                    price_information["ask{}_price".format(i)]
                    * price_information["ask{}_size".format(i)]
                )
        if i == 5 and position > 0:
            # print("the holding could not be bought all clear")
            actual_changed_position = orgional_position - position
        else:
            value += price_information["ask{}_price".format(i)] * position
            actual_changed_position = orgional_position
        # 买的时候相当于多花钱买了
        self.comission_fee_history.append(self.transaction_cost_pct * value)

        return value * (1 + self.transaction_cost_pct), actual_changed_position

    def calculate_value(self, price_information, position):
        return price_information["bid1_price"] * position

    def calculate_avaliable_action(self, price_information):
        # 这块计算跟粒度有关系 修改粒度时应该注意
        buy_size_max = np.sum(
            price_information[["ask1_size", "ask2_size", "ask3_size", "ask4_size"]]
        )
        sell_size_max = np.sum(
            price_information[["bid1_size", "bid2_size", "bid3_size", "bid4_size"]]
        )
        position_upper = self.position + buy_size_max
        position_lower = self.position - sell_size_max
        position_lower = max(position_lower, 0)
        position_upper = min(position_upper, self.max_holding_number)
        # transfer the position back into our action
        current_action = self.position * (self.action_dim - 1) / self.max_holding_number
        action_upper = int(
            position_upper * (self.action_dim - 1) / self.max_holding_number
        )
        if position_lower == 0:
            action_lower = 0
        else:
            action_lower = min(
                int(position_lower * (self.action_dim - 1) / self.max_holding_number)
                + 1,
                action_upper,
                current_action,
            )
        avaliable_discriminator = []
        for i in range(self.action_dim):
            if i >= action_lower and i <= action_upper:
                avaliable_discriminator.append(1)
            else:
                avaliable_discriminator.append(0)
        avaliable_discriminator = torch.tensor(avaliable_discriminator)
        return avaliable_discriminator

    def reset(self):
        # here is a little difference: we only have one asset
        # it starts with the back_num_day and ends in end-self.forward_num_day
        # for the information, it should calculate 2 additional things
        self.terminal = False
        self.day = self.stack_length
        self.data = self.df.iloc[self.day - self.stack_length : self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        price_information = self.data.iloc[-1]
        self.position = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        avaliable_discriminator = self.calculate_avaliable_action(price_information)
        self.previous_position = 0
        self.position = 0
        DP_distribution = [0] * 11
        DP_distribution[self.demonstration_action[self.day - 1]] = 1
        DP_distribution = np.array(DP_distribution)
        self.position_history = []
        # self.first_close = self.data.iloc[-1, :].close

        return self.state.reshape(-1), {
            "previous_action": 0,
            "avaliable_action": avaliable_discriminator,
            "DP_action": DP_distribution,
        }

    def step(self, action):
        # 此处的action应为仓位变化
        normlized_action = action / (self.action_dim - 1)
        position = self.max_holding_number * normlized_action
        # 目前没有future embedding day代表最新一天的信息
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.day += 1
        self.data = self.df.iloc[self.day - self.stack_length : self.day]
        current_price_information = self.data.iloc[-1]
        self.state = self.data[self.tech_indicator_list].values
        self.previous_position = previous_position
        self.position = position
        if previous_position >= position:
            # hold the position or sell some position
            self.sell_size = previous_position - position

            cash, actual_position_change = self.sell_value(
                previous_price_information, self.sell_size
            )
            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            self.position = self.previous_position - actual_position_change
            previous_value = self.calculate_value(
                previous_price_information, self.previous_position
            )
            current_value = self.calculate_value(
                current_price_information, self.position
            )
            self.reward = current_value + cash - previous_value
            # 如果第一开始就是0而且没买
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash - previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)

        if previous_position < position:
            # sell some of the position
            self.buy_size = position - previous_position
            needed_cash, actual_position_change = self.buy_value(
                previous_price_information, self.buy_size
            )
            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)

            self.position = self.previous_position + actual_position_change
            previous_value = self.calculate_value(
                previous_price_information, self.previous_position
            )
            current_value = self.calculate_value(
                current_price_information, self.position
            )
            self.reward = current_value - needed_cash - previous_value
            return_rate = (current_value - needed_cash - previous_value) / (
                previous_value + needed_cash
            )

            self.reward_history.append(self.reward)
            self.return_rate = return_rate
            # print("buy_return_rate", return_rate)
        self.previous_position = self.position
        avaliable_discriminator = self.calculate_avaliable_action(
            current_price_information
        )
        # self.get_final_return_rate()
        # 检查是否出现return rate 为nan的情况
        if self.terminal:
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            (
                return_margin,
                pure_balance,
                required_money,
                commission_fee,
            ) = self.get_final_return_rate()
            self.pured_balance = pure_balance
            self.final_balance = self.pured_balance + self.calculate_value(
                current_price_information, self.position
            )
            self.required_money = required_money
            DP_distribution = [0] * 11

            ##
            # last_day = self.day + 1
            # data = self.df.iloc[last_day -
            #                     self.backward_num_day:last_day, :]
            # last_close = data.iloc[-1, :].close
            # buy_and_hold_profit=100*(last_close-self.first_close)/self.first_close


            tr, sharpe_ratio, vol, mdd, cr, sor = self.evaualte(
                self.save_asset_memoey()
            )
            stats = OrderedDict(
                {
                    "Total Return": ["{:04f}%".format(tr * 100)],
                    # "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                    "Volatility": ["{:04f}%".format(vol* 100)],
                    "Max Drawdown": ["{:04f}%".format(mdd* 100)],
                    # "Calmar Ratio": ["{:04f}".format(cr)],
                    # "Sortino Ratio": ["{:04f}".format(sor)],
                    # "Require Money": ["{:01f}".format(required_money)],
                    # "Commission fee": ["{:01f}".format(commission_fee)],
                    # "Average holding length": ["{:01f}s".format(ahl)],
                }
            )
            table = print_metrics(stats)
            print(table)
            df_value = self.save_asset_memoey()
            daily_return=df_value["daily_return"].values
            assets = df_value["total assets"].values
            ## Excess profit is profit margin
            save_dict = OrderedDict(
                {
                    "Profit Margin": tr * 100,
                    "Excess Profit": tr * 100 - 0,
                    "daily_return": daily_return,
                    "total_assets": assets
                }
            )
            metric_save_path=osp.join(self.work_dir,'metric_'+str(self.task)+'_'+str(self.test_dynamic)+'_'+str(self.test_id)+'_'+str(self.task_index)+'.pickle')
            if self.task == 'test_dynamic':
                with open(metric_save_path, 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print('metric result saved to '+metric_save_path)

        else:
            DP_distribution = [0] * 11
            DP_distribution[self.demonstration_action[self.day - 1]] = 1
            DP_distribution = np.array(DP_distribution)
        self.position_history.append(self.previous_position)

        return (
            self.state.reshape(-1),
            self.reward,
            self.terminal,
            {
                "previous_action": action,
                "avaliable_action": avaliable_discriminator,
                "DP_action": DP_distribution,
            },
        )

    def get_final_return_rate(self, slient=False):
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        balance_list = []
        for i in range(len(true_money)):
            balance_list.append(np.sum(true_money[: i + 1]))
        required_money = -np.min(balance_list)
        commission_fee = np.sum(self.comission_fee_history)

        return (
            final_balance / required_money,
            final_balance,
            required_money,
            commission_fee,
        )

    def save_asset_memoey(self):
        asset_list = [self.required_money]
        for reward in self.reward_history:
            asset_list.append(asset_list[-1] + reward)
        asset_list = asset_list[1:]
        df_value = pd.DataFrame(asset_list)
        df_value.columns = ["total assets"]
        df_value["daily_return"] = self.reward_history
        df_value.index = range(len(df_value))
        return df_value

    def get_daily_return_rate(self,price_list:list):
        return_rate_list=[]
        for i in range(len(price_list)-1):
            return_rate=(price_list[i+1]/price_list[i])-1
            return_rate_list.append(return_rate)
        return return_rate_list
        

    def evaualte(self, df):
        daily_return = df["daily_return"]
        # print(df, df.shape, len(df),len(daily_return))
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1] / (df["total assets"].values[0] + 1e-10) - 1
        return_rate_list=self.get_daily_return_rate(df["total assets"].values)

        sharpe_ratio = np.mean(return_rate_list)*(31536000)** 0.5 / (np.std(return_rate_list)+ 1e-10)
        vol = np.std(return_rate_list)
        mdd = 0
        peak=df["total assets"][0]
        for value in df["total assets"]:
            if value>peak:
                peak=value
            dd=(peak-value)/peak
            if dd>mdd:
                mdd=dd
        cr = np.sum(daily_return) / (mdd + 1e-10)
        sor = np.sum(daily_return) / (np.nan_to_num(np.std(neg_ret_lst),0) + 1e-10) / (np.sqrt(len(daily_return))+1e-10)
        return tr, sharpe_ratio, vol, mdd, cr, sor

    def get_final_return_rate(self, slient=False):
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        balance_list = []
        for i in range(len(true_money)):
            balance_list.append(np.sum(true_money[: i + 1]))
        required_money = -np.min(balance_list)
        commission_fee = np.sum(self.comission_fee_history)

   
        return (
            final_balance / required_money,
            final_balance,
            required_money,
            commission_fee,
        )

    def get_final_return(self):
        return_all = np.sum(self.reward_history)
        return return_all

    def check_sell_needed(self, sell_list, buy_list):
        if len(sell_list) != len(buy_list):
            raise Exception("the dimension is not correct")
        else:
            in_out_list = []
            for i in range(len(sell_list)):
                if sell_list[i] != 0 and buy_list[i] != 0:
                    raise Exception("there is time when money both come in and out")
                elif buy_list[i] != 0 and sell_list[i] != 0:
                    raise Exception("there is time when money both come in and out")
                else:
                    in_out_list.append(sell_list[i] - buy_list[i])
            balance_list = []
            for i in range(len(in_out_list)):
                balance_list.append(np.sum(in_out_list[: i + 1]))
            # print("the money we require is", -min(balance_list))
        return balance_list

    def making_multi_level_dp_demonstration(self, max_punish=1e12):
        # sell_value 和 buy_value与env之间的区别：没有超量一说 一旦超量直接把value打下来
        action_list = []

        def sell_value(price_information, position):
            # use bid price and size to evaluate
            value = 0
            # position 表示剩余的单量
            for i in range(1, 6):
                if position < price_information["bid{}_size".format(i)] or i == 5:
                    break
                else:
                    position -= price_information["bid{}_size".format(i)]
                    value += (
                        price_information["bid{}_price".format(i)]
                        * price_information["bid{}_size".format(i)]
                    )
            if position > 0 and i == 5:
                value = value - max_punish
                # 执行的单量
            else:
                value += price_information["bid{}_price".format(i)] * position
            # 卖的时候的手续费相当于少卖钱了

            return value * (1 - self.transaction_cost_pct)

        def buy_value(price_information, position):
            # this value measure how much
            value = 0
            for i in range(1, 6):
                if position < price_information["ask{}_size".format(i)] or i == 5:
                    break
                else:
                    position -= price_information["ask{}_size".format(i)]
                    value += (
                        price_information["ask{}_price".format(i)]
                        * price_information["ask{}_size".format(i)]
                    )
            if i == 5 and position > 0:
                value = value + max_punish
            else:
                value += price_information["ask{}_price".format(i)] * position
            # 买的时候相当于多花钱买了

            return value * (1 + self.transaction_cost_pct)

        # here we do not consider the level change when the positiion change is too big
        # we do consider the multi granity of our action and max holding case
        scale_factor = self.action_dim - 1

        # init dp solution
        price_information = self.df.iloc[0]
        dp = [[0] * self.action_dim for i in range(len(self.df))]
        for i in range(self.action_dim):
            position_changed = (0 - i) / scale_factor * self.max_holding_number
            if position_changed > 0:
                # 要卖
                dp[0][i] = sell_value(price_information, position_changed)
            else:
                # 要买
                dp[0][i] = -buy_value(price_information, -position_changed)

        for i in range(1, len(self.df)):
            price_information = self.df.iloc[i]
            for j in range(self.action_dim):
                # j是现在的选择
                previous_dp = []
                for k in range(self.action_dim):
                    # k是过去的选择
                    position_changed = (k - j) / scale_factor * self.max_holding_number
                    if position_changed > 0:
                        previous_dp.append(
                            dp[i - 1][k]
                            + sell_value(price_information, position_changed)
                        )
                    else:
                        previous_dp.append(
                            dp[i - 1][k]
                            - buy_value(price_information, -position_changed)
                        )
                dp[i][j] = max(previous_dp)
        # 现在开始倒着取动作
        # 最后一个动作是清仓 看倒数第二个动作是怎么来的
        d1_dp_update = []
        for k in range(self.action_dim):
            position_changed = (k - 0) / scale_factor * self.max_holding_number
            d1_dp_update.append(
                dp[len(self.df) - 2][k]
                + sell_value(price_information, position_changed)
            )
        last_action = d1_dp_update.index(dp[len(self.df) - 1][0])
        last_value = dp[len(self.df) - 2][last_action]
        action_list.append(last_action)
        for i in range(len(self.df) - 2, 0, -1):
            price_information = self.df.iloc[i]
            dn_dp_update = []
            for j in range(self.action_dim):
                position_changed = (
                    (j - last_action) / scale_factor * self.max_holding_number
                )
                if position_changed > 0:
                    dn_dp_update.append(
                        dp[i - 1][j] + sell_value(price_information, position_changed)
                    )
                else:
                    dn_dp_update.append(
                        dp[i - 1][j] - buy_value(price_information, -position_changed)
                    )
            current_action = dn_dp_update.index(last_value)
            last_action = current_action
            last_value = dp[i - 1][last_action]
            action_list.append(last_action)
        action_list.reverse()
        return action_list


@ENVIRONMENTS.register_module()
class HighFrequencyTradingTrainingEnvironment(HighFrequencyTradingEnvironment):
    def __init__(self, **kwargs):
        # super(HighFrequencyTradingTrainingEnvironment, self).__init__()

        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")

        self.df_path = None
        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
        # elif self.task.startswith("valid"):
        #     self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            raise Exception("the training environment is only designed for training")
            # self.df_path = get_attr(self.dataset, "test_path", None)

        self.transaction_cost_pct = get_attr(
            self.dataset, "transaction_cost_pct", 0.00005
        )
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])
        self.stack_length = get_attr(self.dataset, "backward_num_timestamp", 1)
        self.max_holding_number = get_attr(self.dataset, "max_holding_number", 0.01)

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)

        self.action_space = spaces.Discrete(get_attr(self.dataset, "num_action", 11))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list) * self.stack_length,),
        )

        self.action_dim = self.action_space.n
        self.state_dim = self.observation_space.shape[0]
        self.demonstration_action = self.making_multi_level_dp_demonstration(
            max_punish=get_attr(self.dataset, "max_punish", 1e12)
        )
        self.episode_length = get_attr(self.dataset, "episode_length", 14400)
        # reset
        self.terminal = False
        self.day = self.stack_length
        self.data = self.df.iloc[self.day - self.stack_length : self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = 0
        self.position = 0
        self.reward_history = [0]

    def reset(self, i):
        # here is a little difference: we only have one asset
        # it starts with the back_num_day and ends in end-self.forward_num_day
        # for the information, it should calculate 2 additional things
        self.terminal = False
        self.i = i
        self.day = i + self.stack_length
        self.data = self.df.iloc[self.day - self.stack_length : self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        price_information = self.data.iloc[-1]
        self.position = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        avaliable_discriminator = self.calculate_avaliable_action(price_information)
        self.previous_position = 0
        self.position = 0
        DP_distribution = [0] * 11
        DP_distribution[self.demonstration_action[self.day - 1]] = 1
        DP_distribution = np.array(DP_distribution)
        self.position_history = []

        return self.state.reshape(-1), {
            "previous_action": 0,
            "avaliable_action": avaliable_discriminator,
            "DP_action": DP_distribution,
        }

    def step(self, action):
        # 此处的action应为仓位变化
        normlized_action = action / (self.action_dim - 1)
        position = self.max_holding_number * normlized_action
        # 目前没有future embedding day代表最新一天的信息
        self.terminal = (self.day >= len(self.df.index.unique()) - 1) or (
            self.day >= self.i + self.episode_length - 1
        )
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.day += 1
        self.data = self.df.iloc[self.day - self.stack_length : self.day]
        current_price_information = self.data.iloc[-1]
        self.state = self.data[self.tech_indicator_list].values
        self.previous_position = previous_position
        self.position = position
        if previous_position >= position:
            # hold the position or sell some position
            self.sell_size = previous_position - position

            cash, actual_position_change = self.sell_value(
                previous_price_information, self.sell_size
            )
            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            self.position = self.previous_position - actual_position_change
            previous_value = self.calculate_value(
                previous_price_information, self.previous_position
            )
            current_value = self.calculate_value(
                current_price_information, self.position
            )
            self.reward = current_value + cash - previous_value
            # 如果第一开始就是0而且没买
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash - previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)

        if previous_position < position:
            # sell some of the position
            self.buy_size = position - previous_position
            needed_cash, actual_position_change = self.buy_value(
                previous_price_information, self.buy_size
            )
            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)

            self.position = self.previous_position + actual_position_change
            previous_value = self.calculate_value(
                previous_price_information, self.previous_position
            )
            current_value = self.calculate_value(
                current_price_information, self.position
            )
            self.reward = current_value - needed_cash - previous_value
            return_rate = (current_value - needed_cash - previous_value) / (
                previous_value + needed_cash
            )

            self.reward_history.append(self.reward)
            self.return_rate = return_rate
            # print("buy_return_rate", return_rate)
        self.previous_position = self.position
        avaliable_discriminator = self.calculate_avaliable_action(
            current_price_information
        )
        # self.get_final_return_rate()
        # 检查是否出现return rate 为nan的情况
        if self.terminal:
            (
                return_margin,
                pure_balance,
                required_money,
                commission_fee,
            ) = self.get_final_return_rate()
            self.pured_balance = pure_balance
            self.final_balance = self.pured_balance + self.calculate_value(
                current_price_information, self.position
            )
            self.required_money = required_money
            DP_distribution = [0] * 11

            tr, sharpe_ratio, vol, mdd, cr, sor = self.evaualte(
                self.save_asset_memoey()
            )
            stats = OrderedDict(
                {
                    "Profit Margin": ["{:04f}%".format(tr * 100)],
                    "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                    "Volatility": ["{:04f}%".format(vol * 100)],
                    "Max Drawdown": ["{:04f}%".format(mdd * 100)],
                    # "Calmar Ratio": ["{:04f}".format(cr)],
                    # "Sortino Ratio": ["{:04f}".format(sor)],
                    # "Require Money": ["{:01f}".format(required_money)],
                    # "Commission fee": ["{:01f}".format(commission_fee)],
                    # "Average holding length": ["{:01f}s".format(ahl)],
                }
            )
            table = print_metrics(stats)
            print(table)
        else:
            DP_distribution = [0] * 11
            DP_distribution[self.demonstration_action[self.day - 1]] = 1
            DP_distribution = np.array(DP_distribution)
        self.position_history.append(self.previous_position)

        return (
            self.state.reshape(-1),
            self.reward,
            self.terminal,
            {
                "previous_action": action,
                "avaliable_action": avaliable_discriminator,
                "DP_action": DP_distribution,
            },
        )
