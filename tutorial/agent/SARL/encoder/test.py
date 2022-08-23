import argparse
import yaml
import os
import torch
import numpy as np
import random
import sys

sys.path.append("./")

from agent.SARL.encoder.util import prepart_lstm_data, pick_optimizer, LDdataset, dict_to_args
from agent.SARL.model.net import LSTM_clf, m_LSTM_clf
from data.download_data import Dataconfig
from torch.utils.data import DataLoader
from agent.SARL.encoder.train_valid import train_with_valid
import argparse
"""
env 需要重构的东西
state好说 主要是action
state的dimension num_tic*(tech_indicator_list(原来的state)+rank_name(MDD,AR,ER,WR,SR)(咱现在用logical_indicator 预测出来的))+num_tic(上一次的交易[0.1,0.2,...])
action 为discrete的logic_discriptor number. 现在重新构建step中的迭代过程:
有一个函数存portfolio的动作 有一个函数  有一个用网络预测结果的动作 要把之前的df换为tensor
action 选logic_discriptor, 用if映射一个1到5的一个logic_discriptor的选择
选择后用前面定义的evaluate函数得到预测的rank 然后softmax即可
config["tech_indicator_list"]为["high","low","open","close","adjcp",
            "zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10",
            "zd_15", "zd_20", "zd_25", "zd_30"
        ]
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--df_dict",
    type=str,
    default="experiment_result/data/test.csv",
    help="the path for dataframe to generate the portfolio environment")
parser.add_argument(
    "--net_path",
    type=str,
    default="experiment_result/SARL/encoder/best_model/LSTM.pth",
    help="the path for LSTM net")

#where we store the dataset
parser.add_argument("--initial_amount",
                    type=int,
                    default=10000,
                    help="the initial amount")

# where we store the config file
parser.add_argument(
    "--transaction_cost_pct",
    type=float,
    default=0.001,
    help="transaction cost pct",
)

parser.add_argument("--technical_indicator",
                    type=list,
                    default=[
                        "high", "low", "open", "close", "adjcp", "zopen",
                        "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10",
                        "zd_15", "zd_20", "zd_25", "zd_30"
                    ],
                    help="the name of the features to predict the label")
parser.add_argument(
    "--num_day",
    type=int,
    default=5,
    help="the number of day",
)

args = parser.parse_args()
print(vars(args))
with open(
        '/home/sunshuo/qml/TradeMaster_reframe/input_config/env/portfolio/portfolio_for_SARL/test.yml',
        'w') as f:
    data = yaml.dump(vars(args), f)