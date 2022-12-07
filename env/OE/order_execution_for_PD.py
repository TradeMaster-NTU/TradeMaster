from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
# notice that although the title contains OE, the state does not contain any information about order
# but traditional OHLCV like algorithm trading
# the key difference of this OE and algorithm trading is that it can only put action at one side, i.e. if
#the OE task is asked to buy one share of BTC, you can not put ask order even you still have BTC in your hand
# therefore, we use the dataset of BTC algorithm trading yet the env is a little different.
# in oe, we want to sell at the highest or buy at the lowest, therefore the optimization target is set to be
# the amount of money we sell, if the target is buying, our target will be nagative, they will both be optimized
# to their max
# all of the trades will be conducted at their close price
# this give me a little bit confusion about the sparation way of dataset, because it is not like the
#discrete action space where we can adjust the target amount with portion to the that of a train dataset
# and the valid dataset, which might cause a problem. Here to be more speicifcally
parser.add_argument(
    "--df_path",
    type=str,
    default="data/data/BTC/valid.csv",
    help="the path for the downloaded data to generate the environment")
parser.add_argument("--tech_indicator_list",
                    type=list,
                    default=[
                        "high", "low", "open", "close", "adjcp", "zopen",
                        "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10",
                        "zd_15", "zd_20", "zd_25", "zd_30"
                    ],
                    help="the name of the features to predict the label")
parser.add_argument("--sell_target",
                    type=int,
                    default=1,
                    choices=[-1, 1],
                    help="1 for sell and -1 for buy")
parser.add_argument("--num_back_day",
                    type=int,
                    default=10,
                    help="number of backwards day")


class TradingEnv(gym.Env):
    def __init__(self, config):
        #init
        self.tech_indicator_list = config["tech_indicator_list"]
        self.df = pd.read_csv(config["df_path"], index_col=0)
        self.action_target = config["sell_target"]
        self.backward_num_day = config["num_back_day"]
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1, ),
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.backward_num_day, len(self.tech_indicator_list)),
        )
        # reset

        # the compound_memory's element consists of 2 parts: the cash and the number of bitcoin you have in hand

        self.day = self.backward_num_day
        self.data_public_imperfect = self.df.iloc[
            self.day - self.backward_num_day:self.day, :]
        self.data_public_perfect = self.df.iloc[
            self.day - self.backward_num_day:self.day +
            self.backward_num_day, :]
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
        self.private_state_list = [self.private_state] * self.backward_num_day

    def reset(self):
        self.terminal = False
        self.day = self.backward_num_day
        self.data_public_imperfect = self.df.iloc[
            self.day - self.backward_num_day:self.day, :]
        self.data_public_perfect = self.df.iloc[
            self.day - self.backward_num_day:self.day +
            self.backward_num_day, :]
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
        self.private_state_list = [self.private_state] * self.backward_num_day
        return np.array(self.public_imperfect_state), {
            "perfect_state": np.array(self.public_perfect_state),
            "private_state": np.array([self.private_state_list])
        }

    def step(self, action):
        # based on the current price information, we decided whether to trade use the next day's price
        # the reward is calculated as at*(p_(t+1)-average(p))
        self.day = self.day + 1

        self.terminal = (self.day >= (len(self.df) - self.backward_num_day))
        if self.terminal:
            leftover_day, leftover_order = self.private_state
            print("done")
            self.data_public_imperfect = self.df.iloc[
                self.day - self.backward_num_day:self.day, :]
            current_price = self.data_public_imperfect.iloc[-1].close
            self.money_sold += leftover_order * current_price
            self.public_imperfect_state = np.array(self.public_imperfect_state)
            self.private_state_list.append([0, 0])
            self.private_state_list.remove(self.private_state_list[0])

            return self.public_imperfect_state, self.reward, self.terminal, {
                "perfect_state": np.array([self.public_perfect_state]),
                "private_state": np.array([self.private_state_list])
            }

        else:
            leftover_day, leftover_order = self.private_state

            previous_average_price = np.mean(self.df.iloc[:self.day -
                                                          1].close.values)
            self.data_public_imperfect = self.df.iloc[
                self.day - self.backward_num_day:self.day, :]
            self.data_public_perfect = self.df.iloc[
                self.day - self.backward_num_day:self.day +
                self.backward_num_day, :]
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
                self.terminal = True
                print("done")
            leftover_day, leftover_order = leftover_day - 1 / (len(
                self.df) - 2 * self.backward_num_day), leftover_order - action
            self.private_state = np.array([leftover_day, leftover_order])
            self.private_state_list.append(self.private_state)
            self.private_state_list.remove(self.private_state_list[0])
            return self.public_imperfect_state, self.reward, self.terminal, {
                "perfect_state": np.array([self.public_perfect_state]),
                "private_state": np.array([self.private_state_list])
            }

    def find_money_sold(self):
        return self.money_sold


if __name__ == "__main__":
    args = parser.parse_args()
    config = vars(args)
    env = TradingEnv(config)
    print(env.observation_space.shape)
    _, info = env.reset()
    print(info["private_state"].shape)
    # action = 0.00001
    # done = False
    # i = 0
    # while not done:
    #     s, r, done, info = env.step(action)
    #     print(s.shape)
    #     print(info["private_state"].shape)
    # print(env.money_sold)
    # print(i)
    # import yaml
    # # args = parser.parse_args()

    # def save_dict_to_yaml(dict_value: dict, save_path: str):
    #     with open(save_path, 'w') as file:
    #         file.write(yaml.dump(dict_value, allow_unicode=True))

    # def read_yaml_to_dict(yaml_path: str, ):
    #     with open(yaml_path) as file:
    #         dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
    #         return dict_value

    # save_dict_to_yaml(
    #     vars(args),
    #     "/home/sunshuo/qml/TradeMaster-1/config/input_config/env/OE/OE_for_PD/valid.yml"
    # )