import os
import yaml
import torch
import sys

sys.path.append(".")
from agent.Investor_Imitator.model.net import MLP_reg, MLP_cls

import argparse

parser = argparse.ArgumentParser()

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

args = parser.parse_args()

a = vars(args)
print(a)
new_path = "config/input_config/agent/investor_imitator"
print(a)
if not os.path.exists(new_path):
    os.makedirs(new_path)
with open(new_path + "/RL.yml", "w", encoding="utf-8") as f:
    yaml.dump(a, f)