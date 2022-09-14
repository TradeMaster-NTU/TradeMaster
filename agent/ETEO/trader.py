from logging import raiseExceptions
from re import L
import sys

sys.path.append(".")
from agent.ETEO.model import FCN_stack_ETTO, LSTM_ETEO
from agent.ETEO.util import set_seed, load_yaml
from env.OE.order_execution_for_ETEO import TradingEnv
import torch
import os
import argparse
from torch import nn
import random
from torch.distributions import Normal
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed",
                    type=int,
                    default=12345,
                    help="the value of the random seed")
parser.add_argument("--env_config_path",
                    type=str,
                    default="config/input_config/env/OE/OE_for_ETEO/",
                    help="the path for storing the config file for deeptrader")
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="learning rate",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="result/ETEO/trained_model",
    help="the path for trained model",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/ETEO/test_result",
    help="the path for test result",
)
parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="the number of epoch we train",
)
parser.add_argument(
    "--net_category",
    type=str,
    default="stacked",
    choices=["stacked", "lstm"],
    help="the name of the category of the net we use for v and action",
)
parser.add_argument(
    "--lenth_state",
    type=int,
    default="10",
    help=
    "the length of the state, ie the number of timestamp that contains in the input of the net",
)


class trader:
    def __init__(self, args) -> None:
        self.seed = args.random_seed
        set_seed(self.seed)
        self.num_epoch = args.num_epoch
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cpu' if self.GPU_IN_USE else 'cpu')
        self.model_path = args.model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.result_path = args.result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.train_env_config = load_yaml(args.env_config_path + "train.yml")
        self.valid_env_config = load_yaml(args.env_config_path + "valid.yml")
        self.test_env_config = load_yaml(args.env_config_path + "test.yml")
        self.train_env_instance = TradingEnv(self.train_env_config)
        self.valid_env_instance = TradingEnv(self.valid_env_config)
        self.test_env_instance = TradingEnv(self.test_env_config)
        self.num_features = self.train_env_instance.observation_space.shape[0]
        self.net_category = args.net_category
        # 两套网络（新与旧 来对比计算重采样的大小以及此次更新的大小尺度）
        # 由于两种网络的输入不同 因此我们这里目前只写stacked版本的更新
        if args.net_category not in ["stacked", "lstm"]:
            raiseExceptions(
                "we haven't implement that kind of net, please choose stacked or lstm"
            )
        if args.net_category == "stacked":
            self.net_old = FCN_stack_ETTO(args.lenth_state, self.num_features)
            self.net_new = FCN_stack_ETTO(args.lenth_state, self.num_features)
        if args.net_category == "lstm":
            self.net_old = LSTM_ETEO(args.lenth_state, self.num_features)
            self.net_new = LSTM_ETEO(args.lenth_state, self.num_features)
