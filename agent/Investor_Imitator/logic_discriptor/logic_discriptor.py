from re import A
import sys

from scipy import ifft

sys.path.append("./")
from data.download_data import Dataconfig
from agent.Investor_Imitator.logic_discriptor.utli import make_label_tic, evaluate, \
    compare_average, make_rank_label, LDdataset, pick_optimizer,dict_to_args
from agent.Investor_Imitator.logic_discriptor.train_valid import train_with_valid
from agent.Investor_Imitator.model.net import MLP_reg
from ast import Raise
from importlib.resources import path
from logging import raiseExceptions
from operator import index
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

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
                    default="./data/data/",
                    help="the path for storing the downloaded data")

#where we store the dataset
parser.add_argument(
    "--output_config_path",
    type=str,
    default="config/output_config/data",
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
    default="config/input_config/data/custom.yml",
    help=
    "determine the location of a yaml file used to initialize the Dataconfig Class"
)

parser.add_argument(
    "--discriptor_path",
    type=str,
    default="./result/invest_imitator/logic_discriptor",
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
    default="config/input_config/agent/investor_imitator/logic_discriptor.yml",
    help=
    "determine the path of a yaml file as the overall input of the logic_discriptor"
)
args = parser.parse_args()


#TODO test the generation process
class logic_discriptor:
    def __init__(self, args):
        if args.input_logic_config == False:
            self.dataconfig = Dataconfig(args)
            self.tic_list = self.dataconfig.train_dataset.tic.unique()
            self.discriptor_path = args.discriptor_path
            self.make_dict()
            self.seed = args.seed
            self.set_seed()
            self.rank_name_list = args.rank_name_list
            self.batch_size = args.batch_size
            self.hidden_size = args.hidden_size
            self.num_epoch = args.num_epoch
            self.optimizer = pick_optimizer(args.optimizer)
            self.df_train_list, self.AR_train_list, self.MDD_train_list, self.SR_train_list, self.ER_train_list,\
             self.WR_train_list = make_label_tic(
            args.num_day, self.dataconfig.train_dataset)
            self.df_valid_list, self.AR_valid_list, self.MDD_valid_list, self.SR_valid_list,self.ER_valid_list,\
             self.WR_valid_list = make_label_tic(
            args.num_day, self.dataconfig.valid_dataset)
        else:
            with open(args.input_logic_config_dict, "r") as file:
                self.config = yaml.safe_load(file)
            args = dict_to_args(self.config)
            self.dataconfig = Dataconfig(args)
            self.tic_list = self.dataconfig.train_dataset.tic.unique()
            self.discriptor_path = self.config["discriptor_path"]
            self.make_dict()
            self.seed = self.config["seed"]
            self.set_seed()
            self.rank_name_list = self.config["rank_name_list"]
            self.batch_size = self.config["batch_size"]
            self.hidden_size = self.config["hidden_size"]
            self.num_epoch = self.config["num_epoch"]
            self.optimizer = pick_optimizer(self.config["optimizer"])
            self.df_train_list, self.AR_train_list, self.MDD_train_list, self.SR_train_list, self.ER_train_list,\
             self.WR_train_list = make_label_tic(
            self.config["num_day"], self.dataconfig.train_dataset)
            self.df_valid_list, self.AR_valid_list, self.MDD_valid_list, self.SR_valid_list,self.ER_valid_list,\
             self.WR_valid_list = make_label_tic(
            self.config["num_day"], self.dataconfig.valid_dataset)


        self.AR_train_list, self.MDD_train_list, self.SR_train_list, self.ER_train_list,\
             self.WR_train_list = np.array(
            self.AR_train_list), np.array(self.MDD_train_list), np.array(
                self.SR_train_list),np.array(self.ER_train_list),np.array(self.WR_train_list)

        self.AR_valid_list, self.MDD_valid_list, self.SR_valid_list,self.ER_valid_list,\
            self.WR_valid_list = np.array(
            self.AR_valid_list), np.array(self.MDD_valid_list), np.array(
                self.SR_valid_list),np.array(self.ER_valid_list),np.array(self.WR_valid_list)


        self.AR_train_list, self.MDD_train_list, self.SR_train_list,self.ER_train_list,\
            self.WR_train_list = make_rank_label(
            self.AR_train_list), make_rank_label(
                self.MDD_train_list,
                reverse=True),make_rank_label(self.SR_train_list),\
                    make_rank_label(self.ER_train_list),make_rank_label(self.WR_train_list)



        self.AR_valid_list, self.MDD_valid_list, self.SR_valid_list,self.ER_valid_list,\
            self.WR_valid_list =make_rank_label(
            self.AR_valid_list), make_rank_label(
                self.MDD_valid_list,
                reverse=True), make_rank_label(self.SR_valid_list),make_rank_label(self.ER_valid_list),\
                    make_rank_label(self.WR_valid_list)

        self.prepare()

    def make_dict(self):
        if not os.path.exists(self.discriptor_path):
            os.makedirs(self.discriptor_path)

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.seed = self.seed

    def prepare(self):
        for rank_name in self.rank_name_list:
            if rank_name not in ["AR", "SR", "MDD", "ER", "WR"]:
                raiseExceptions(
                    "we do not support the indicator {} yet".format(rank_name))
            elif rank_name == "AR":
                self.label_train = self.AR_train_list
                self.label_valid = self.AR_valid_list
            elif rank_name == "SR":
                self.label_train = self.SR_train_list
                self.label_valid = self.SR_valid_list
            elif rank_name == "MDD":
                self.label_train = self.MDD_train_list
                self.label_valid = self.MDD_valid_list
            elif rank_name == "ER":
                self.label_train = self.ER_train_list
                self.label_valid = self.ER_valid_list
            elif rank_name == "WR":
                self.label_train = self.WR_train_list
                self.label_valid = self.WR_valid_list
            self.discriptor_indicator_path = self.discriptor_path + "/" + rank_name
            self.net = MLP_reg(input_size=self.df_train_list[0].shape[1],
                               hidden_size=self.hidden_size)
            self.train_dataset = LDdataset(self.df_train_list,
                                           self.label_train)
            self.valid_dataset = LDdataset(self.df_valid_list,
                                           self.label_valid)
            self.train_dataLoader = DataLoader(self.train_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True)
            self.valid_dataLoader = DataLoader(self.valid_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=False)
            self.optimizer_instance = self.optimizer(self.net.parameters())
            train_with_valid(self.train_dataLoader, self.valid_dataLoader,
                             self.num_epoch, self.net, self.optimizer_instance,
                             self.tic_list, self.discriptor_indicator_path)


if __name__ == "__main__":
    a = logic_discriptor(args)
