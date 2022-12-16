import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import yaml
# TODO df略有不同 需要有一列vadility来进行计算
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
# from stable_baselines3.common.env_checker import check_envs

parser.add_argument(
    "--df_path",
    type=str,
    default="./data/data/BTC_for_iRDPG/train.csv",
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
                        'normal_open', 'normal_close', 'normal_high',
                        'normal_low', 'normal_EMA_12_close', 'normal_EMA_26_close',
                        'normal_EMA_10_close', 'normal_BB_mid', 'normal_BB_low',
                        'normal_BB_high', 'normal_MACD'
                    ],
                    help="the name of the features to predict the label")

parser.add_argument(
    "--backward_num_day",
    type=int,
    default=14,
    help="the number of day to calculate the variance of the assets ",
)


class TradingEnv(gym.Env):
    def __init__(self, config):
        # init
        self.initial_amount = config["initial_amount"]
        self.transaction_cost_pct = config["transaction_cost_pct"]
        self.tech_indicator_list = config["tech_indicator_list"]
        self.backward_num_day = config["backward_num_day"]
        self.df = pd.read_csv(config["df_path"], index_col=0)
        self.money_at_hand = self.initial_amount

        self.action_space = spaces.Discrete(2)
        # long or short
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.backward_num_day, len(self.tech_indicator_list)),
        )
        # reset
        self.position = 0
        # the compound_memory's element consists of 2 parts: the cash and the number of bitcoin you have in hand
        self.infos = []
        self.terminal = False
        self.stepIdx = 0
        self.profit = 0
        self.hold_float = 0
        self.step_hold_profit = 0
        self.data=self.df.iloc[self.stepIdx:self.stepIdx +
                                  self.backward_num_day]
        self.state = self.df.iloc[self.stepIdx:self.stepIdx +
                                  self.backward_num_day][self.tech_indicator_list].values
        
        self.return_memory = [0]

    def reset(self):
        self.position = 0
        # the compound_memory's element consists of 2 parts: the cash and the number of bitcoin you have in hand
        self.infos = []
        self.terminal = False
        self.stepIdx = 0
        self.profit = 0
        self.hold_float = 0
        self.step_hold_profit = 0
        self.data=self.df.iloc[self.stepIdx:self.stepIdx +
                                  self.backward_num_day]
        self.state = self.df.iloc[self.stepIdx:self.stepIdx +
                                  self.backward_num_day][self.tech_indicator_list].values
        self.return_memory = [0]
        return self.state

    def step(self, action):
        action = 2*(action-1/2)
        self.terminal = (self.stepIdx >= len(self.df)-self.backward_num_day-1)
        if self.terminal:
            self.stepIdx += 1
            self.data=self.df.iloc[self.stepIdx:self.stepIdx +
                                  self.backward_num_day]
            self.state = self.df.iloc[self.stepIdx:self.stepIdx +
                                      self.backward_num_day][self.tech_indicator_list].values
            self.reward = 0
            self.bc_action = 0
            return self.state, self.reward, self.terminal, self.bc_action
        else:
            self.stepIdx += 1
            self.data=self.df.iloc[self.stepIdx:self.stepIdx +
                                  self.backward_num_day]
            self.state = self.df.iloc[self.stepIdx:self.stepIdx +
                                      self.backward_num_day][self.tech_indicator_list].values
            self.bc_action = self.df.iloc[self.stepIdx:self.stepIdx +
                                          self.backward_num_day].iloc[-1].best_action
            self.trade(action)
            self.DSR_reward()

            return self.state, self.reward, self.terminal, self.bc_action

    def trade(self, action):
        if action == 0:
            print('trading_action is zero, which is wrong.')
            raise AssertionError('trading_action is zero')
        pt0 = self.data["close"].iloc[-2]
        pt1 = self.data["close"].iloc[-1]
        self.money_at_hand_now = (action*(pt1-pt0)/pt0+1)*self.money_at_hand - \
            self.transaction_cost_pct*self.money_at_hand * \
            (np.abs(action-self.position))
        self.profit = self.money_at_hand_now-self.money_at_hand
        self.money_at_hand = self.money_at_hand_now

    def DSR_reward(self):
        sr0 = np.mean(self.return_memory)/np.std(self.return_memory)
        self.return_memory.append(self.profit)
        sr1 = np.mean(self.return_memory)/np.std(self.return_memory)
        self.reward = sr1-sr0
if __name__=="__main__":
    import yaml

    args = parser.parse_args()

    # env = TradingEnv(vars(args))
    # done=False
    # state = env.reset()
    # action=1
    # while not done:
    #     state, reward, done, bc_action=env.step(action)
    #     print(bc_action)
    
    # args = parser.parse_args()

    def save_dict_to_yaml(dict_value: dict, save_path: str):
        with open(save_path, 'w') as file:
            file.write(yaml.dump(dict_value, allow_unicode=True))

    def read_yaml_to_dict(yaml_path: str, ):
        with open(yaml_path) as file:
            dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
            return dict_value

    save_dict_to_yaml(
        vars(args),
        "/home/sunshuo/qml/TradeMaster-1/config/input_config/env/AT/AT_for_iRDPG/train.yml"
    )
