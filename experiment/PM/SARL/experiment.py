import argparse
import yaml
import os
import torch
import numpy as np
import random
import sys

sys.path.append("./")
from agent.SARL.encoder.encoder import encoder
from agent.SARL.encoder.util import prepart_lstm_data, pick_optimizer, LDdataset, dict_to_args, prepart_m_lstm_data, m_lstm_dataset
from agent.SARL.model.net import LSTM_clf, m_LSTM_clf
from data.download_data import Dataconfig
from torch.utils.data import DataLoader
from agent.SARL.encoder.train_valid import train_with_valid
from ray.tune.registry import register_env
import ray
import pandas as pd
import os
from agent.SARL.RL.trader import env_creator, load_yaml, select_algorithms, agent

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",
                    type=str,
                    default="./experiment_result/data/",
                    help="the path for storing the downloaded data")

#where we store the dataset
parser.add_argument(
    "--output_config_path",
    type=str,
    default="./output_config/data",
    help="the path for storing the generated config file for data")

# where we store the config file
parser.add_argument(
    "--dataset",
    choices=["dj30", "sz50", "acl18", "futures", "crypto", "exchange"],
    default="dj30",
    help="the name of the dataset",
)

parser.add_argument("--split_proportion",
                    type=list,
                    default=[0.8, 0.1, 0.1],
                    help="the split proportion for train, valid and test")

parser.add_argument(
    "--generate_config",
    type=bool,
    default=True,
    help=
    "determine whether to generate a yaml file to memorize the train valid and test'data's dict"
)

parser.add_argument(
    "--input_config",
    type=bool,
    default=False,
    help=
    "determine whether to use a yaml file as the overall input of the Dataconfig, this is needed when have other format of dataset"
)

parser.add_argument(
    "--input_config_path",
    type=str,
    default="input_config/data/custom.yml",
    help=
    "determine the location of a yaml file used to initialize the Dataconfig Class"
)

parser.add_argument(
    "--encoder_path",
    type=str,
    default="./experiment_result/SARL/encoder",
    help="the path for storing the generated encoder file for data")

parser.add_argument("--seed",
                    type=int,
                    default=12345,
                    help="the random seed to train the logic discriptor")

parser.add_argument(
    "--num_day",
    type=int,
    default=5,
    help="the number of the day for us to use to predict the label")

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="the batch size of the data during the training process")

parser.add_argument("--hidden_size",
                    type=int,
                    default=128,
                    help="the size of the hidden nodes of MLP_reg ")

parser.add_argument(
    "--optimizer",
    choices=[
        "Adam", "SGD", "ASGD", "Rprop", "Adagrad", "Adadelta", "RMSprop",
        "Adamax", "SparseAdam", "LBFGS"
    ],
    default="Adam",
    help="the name of the optimizer",
)

parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="the number of epoch",
)
parser.add_argument(
    "--input_encoder_config",
    type=bool,
    default=False,
    help=
    "determine whether to use a yaml file as the overall input of the logic_discriptor, this is needed when have other format of dataset"
)

parser.add_argument(
    "--input_encoder_config_dict",
    type=str,
    default="input_config/agent/SARL/encoder.yml",
    help=
    "determine the path of a yaml file as the overall input of the logic_discriptor"
)

parser.add_argument("--technical_indicator",
                    type=list,
                    default=[
                        "high", "low", "open", "close", "adjcp", "zopen",
                        "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10",
                        "zd_15", "zd_20", "zd_25", "zd_30"
                    ],
                    help="the name of the features to predict the label")
parser.add_argument("--num_layer",
                    type=int,
                    default=1,
                    help="the number of layer in the LSTM")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="the learning rate for encoder")

parser.add_argument(
    "--env_name",
    choices=["portfolio"],
    default="portfolio",
    help="the name of TradingEnv ",
)
parser.add_argument(
    "--dict_trained_model",
    default="experiment_result/SARL/trained_model/",
    help="the dict of the trained model ",
)

parser.add_argument(
    "--train_env_config_dict",
    default="input_config/env/portfolio/portfolio_for_SARL/train.yml",
    help="the dict of the train config of TradingEnv ",
)

parser.add_argument(
    "--valid_env_config_dict",
    default="input_config/env/portfolio/portfolio_for_SARL/valid.yml",
    help="the dict of the valid config of TradingEnv ",
)

parser.add_argument(
    "--test_env_config_dict",
    default="input_config/env/portfolio/portfolio_for_SARL/test.yml",
    help="the dict of the test config of TradingEnv ",
)

parser.add_argument(
    "--name_of_algorithms",
    choices=["PPO", "A2C", "SAC", "TD3", "PG", "DDPG"],
    type=str,
    default="DDPG",
    help="name_of_algorithms ",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="the number of training epoch",
)

parser.add_argument(
    "--random_seed",
    type=int,
    default=12345,
    help="the number of training epoch",
)

parser.add_argument(
    "--model_config_dict",
    type=str,
    default="input_config/agent/SOTA/DDPG.yml",
    help="the dict of the model_config file",
)

parser.add_argument(
    "--result_dict",
    type=str,
    default="experiment_result/SARL/test_result/",
    help="the dict of the result of the test",
)

args = parser.parse_args()


def get_encoder(args):
    encoder(args)


def get_agent(args):
    trader = agent(args)
    return trader


def experiment(agent: agent):
    agent.train_with_valid()
    agent.test()


def main(args):
    get_encoder(args)
    agent = get_agent(args)
    experiment(agent)


if __name__ == '__main__':
    main(args)