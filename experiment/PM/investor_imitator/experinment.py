from re import A
import sys

from scipy import ifft

sys.path.append("./")
from data.download_data import Dataconfig
from agent.Investor_Imitator.logic_discriptor.utli import make_label_tic, evaluate, \
    compare_average, make_rank_label, LDdataset, pick_optimizer,dict_to_args
from agent.Investor_Imitator.logic_discriptor.train_valid import train_with_valid
from agent.Investor_Imitator.model.net import MLP_reg, MLP_cls
from agent.Investor_Imitator.RL.RL import Agent, env_creator, load_yaml
from ast import Raise
from importlib.resources import path
from logging import raiseExceptions
from operator import index
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from agent.Investor_Imitator.logic_discriptor.logic_discriptor import logic_discriptor
import numpy
import os
import pandas as pd
import data.preprocess as p
import sys
import argparse
import yaml
# basic setting for dataset choice and the dicts you want to store
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
    "--discriptor_path",
    type=str,
    default="./experiment_result/invest_imitator/logic_discriptor",
    help="the path for storing the generated logic_discriptor file for data")

parser.add_argument("--seed",
                    type=int,
                    default=12345,
                    help="the random seed to train the logic discriptor")

parser.add_argument("--num_day",
                    type=int,
                    default=10,
                    help="the number of the day for us to make the label")

parser.add_argument(
    "--rank_name_list",
    type=list,
    default=["AR", "SR", "MDD", "ER", "WR"],
    help=
    "the list of the name of indicator we use to train the logic discriptor")

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="the batch size of the data during the training process")

parser.add_argument("--hidden_size",
                    type=int,
                    default=256,
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
    "--input_logic_config",
    type=bool,
    default=False,
    help=
    "determine whether to use a yaml file as the overall input of the logic_discriptor, this is needed when have other format of dataset"
)
parser.add_argument(
    "--input_logic_config_dict",
    type=str,
    default="input_config/agent/investor_imitator/logic_discriptor.yml",
    help=
    "determine the path of a yaml file as the overall input of the logic_discriptor"
)

parser.add_argument(
    "--env_name",
    choices=["portfolio"],
    default="portfolio",
    help="the name of TradingEnv ",
)
parser.add_argument(
    "--dict_trained_model",
    default="experiment_result/invest_imitator/trained_model",
    help="the dict of the trained model ",
)

parser.add_argument(
    "--train_env_config_dict",
    default=
    "input_config/env/portfolio/portfolio_for_investor_imitator/train.yml",
    help="the dict of the train config of TradingEnv ",
)

parser.add_argument(
    "--valid_env_config_dict",
    default=
    "input_config/env/portfolio/portfolio_for_investor_imitator/valid.yml",
    help="the dict of the valid config of TradingEnv ",
)

parser.add_argument(
    "--test_env_config_dict",
    default=
    "input_config/env/portfolio/portfolio_for_investor_imitator/test.yml",
    help="the dict of the test config of TradingEnv ",
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
    "--result_dict",
    type=str,
    default="experiment_result/invest_imitator/test_result/",
    help="the dict of the result of the test",
)
parser.add_argument(
    "--input_config_RL",
    type=bool,
    default=False,
    help="whether to use yaml as the config",
)
parser.add_argument(
    "--input_config_dict_RL",
    type=str,
    default="input_config/agent/investor_imitator/RL.yml",
    help="the dict of yaml",
)

args = parser.parse_args()


def get_discriptor(args):
    logic_discriptor(args)


def get_agent(args):
    agent = Agent(args)
    return agent


def experiment(agent: Agent):
    agent.train_with_valid()
    agent.test()


def main(args):
    get_discriptor(args)
    agent = get_agent(args)
    experiment(agent)


if __name__ == '__main__':
    main(args)