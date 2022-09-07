import sys

sys.path.append(".")
from agent.DeepTrader.data.market_information import make_market_information, make_correlation_information
from agent.DeepTrader.model.model import Chomp1d, TemporalBlock, TemporalConvNet, SA, GCN, IN, IN_value, asset_scoring, asset_scoring_value, market_scoring
from agent.DeepTrader.model.portfolio_generator import generate_portfolio, generate_rho
from agent.DeepTrader.util.utli import set_seed, load_yaml
from env.PM.portfolio_for_deeptrader import Tradingenv
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed",
                    type=int,
                    default=12345,
                    help="the value of the random seed")
parser.add_argument(
    "--env_config_path",
    type=str,
    default="config/input_config/env/portfolio/portfolio_for_deeptrader/",
    help="the path for storing the config file for deeptrader")

parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="the gamma for DPG",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="result/DeepTrader/trained_model",
    help="the path for trained model",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DeepTrader/test_result",
    help="the path for test result",
)
parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="the number of epoch we train",
)


class trader:
    def __init__(self, args) -> None:
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
        self.train_env_instance = Tradingenv(self.train_env_config)
        self.valid_env_instance = Tradingenv(self.valid_env_config)
        self.test_env_instance = Tradingenv(self.test_env_config)
        self.day_length = self.train_env_config["length_day"]
        self.input_channel = len(self.train_env_config["tech_indicator_list"])
        self.asset_policy = asset_scoring(N=self.train_env_instance.stock_dim,
                                          K_l=self.day_length,
                                          num_inputs=self.input_channel,
                                          num_channels=[12, 12, 12])
        self.asset_policy_critic = asset_scoring_value(
            N=self.train_env_instance.stock_dim,
            K_l=self.day_length,
            num_inputs=self.input_channel,
            num_channels=[12, 12, 12])
        self.market_policy = market_scoring(self.input_channel)
        self.optimizer_asset_actor = torch.optim.Adam(
            self.asset_policy.parameters(), lr=1e-4)
        self.optimizer_asset_critic = torch.optim.Adam(
            self.asset_policy_critic.parameters(), lr=1e-4)
        self.optimizer_market_policy = torch.optim.Adam(
            self.market_policy.parameters(), lr=1e-4)
