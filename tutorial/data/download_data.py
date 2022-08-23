from ast import Raise
from importlib.resources import path
from logging import raiseExceptions
from operator import index
import sys

sys.path.append("./")
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
    default="./config/output_config/data",
    help="the path for storing the generated config file for data")
# where we store the config file
parser.add_argument(
    "--dataset",
    choices=["dj30", "sz50", "acl18", "futures", "crypto", "exchange", "BTC"],
    default="BTC",
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


#TODO complete the generate_config and input_config function
class Dataconfig:
    def __init__(self, args):
        if args.input_config == False:
            self.data_path = args.data_path
            self.output_config_path = args.output_config_path
            self.data = args.dataset
            self.proportion = args.split_proportion
            self.generate_config = args.generate_config
            self.make_dict()
            self.download_split_data()
            self.generate_yaml()
        else:
            with open(args.input_config_path, "r") as file:
                self.config = yaml.safe_load(file)
            print(self.config)
            self.data_path = self.config["data_path"]
            self.data = self.config["data"]
            self.proportion = self.config["split_proportion"]
            self.generate_config = self.config["generate_config"]
            self.datasource = self.config["datasource"]
            self.output_config_path = self.config["output_config_path"]
            self.make_dict_custom()
            self.download_split_data_custom()
            self.generate_yaml()

    def make_dict(self):
        if (self.data in [
                "dj30", "sz50", "acl18", "futures", "crypto", "exchange", "BTC"
        ]) == False:
            raiseExceptions(
                "this dataset is not supported yet, you can change the custom.yaml and\
                 make input_config True to custom this model")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.data_dict = self.data_path + self.data
        if not os.path.exists(self.data_dict):
            os.makedirs(self.data_dict)

    def make_dict_custom(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.data_dict = self.data_path + self.data
        if not os.path.exists(self.data_dict):
            os.makedirs(self.data_dict)

    def download_split_data(self):
        datasource = " https://raw.githubusercontent.com/qinmoelei/TradeMater-Data/main/"
        command = "wget -P"+self.data_dict +\
            datasource+self.data+".csv"
        if not os.path.exists(self.data_dict + "/" + self.data + ".csv"):
            print(command)
            os.system(command)
        self.whole_data_dict = self.data_dict + "/" + self.data + ".csv"
        self.train_data_dict = self.data_dict + "/" + "train.csv"
        self.valid_data_dict = self.data_dict + "/" + "valid.csv"
        self.test_data_dict = self.data_dict + "/" + "test.csv"

        self.dataset = pd.read_csv(self.data_dict + "/" + self.data + ".csv",
                                   index_col=0)
        self.dataset = p.generate_normalized_feature(self.dataset)
        train, valid, test = p.split(self.dataset, self.proportion)
        self.train_dataset, self.valid_dataset, self.test_dataset = train, valid, test
        self.dataset.to_csv(self.whole_data_dict)
        self.train_dataset.to_csv(self.train_data_dict)
        self.valid_dataset.to_csv(self.valid_data_dict)
        self.test_dataset.to_csv(self.test_data_dict)

    def download_split_data_custom(self):
        # notice that the downloaded file name should share the same name as self.data
        command = "wget -P" + self.data_dict + " " + self.datasource
        if not os.path.exists(self.data_dict + "/" + self.data + ".csv"):
            os.system(command)
        self.whole_data_dict = self.data_dict + "/" + self.data + ".csv"
        self.train_data_dict = self.data_dict + "/" + "train.csv"
        self.valid_data_dict = self.data_dict + "/" + "valid.csv"
        self.test_data_dict = self.data_dict + "/" + "test.csv"

        self.dataset = pd.read_csv(self.data_dict + "/" + self.data + ".csv",
                                   index_col=0)
        self.dataset = p.generate_normalized_feature(self.dataset)
        train, valid, test = p.split(self.dataset, self.proportion)
        self.train_dataset, self.valid_dataset, self.test_dataset = train, valid, test
        self.dataset.to_csv(self.whole_data_dict)
        self.train_dataset.to_csv(self.train_data_dict)
        self.valid_dataset.to_csv(self.valid_data_dict)
        self.test_dataset.to_csv(self.test_data_dict)

    def generate_yaml(self):
        if self.generate_config == True:
            config = dict(
                train_data_dict=self.train_data_dict,
                valid_data_dict=self.valid_data_dict,
                test_data_dict=self.test_data_dict,
            )
            data_config_path = self.output_config_path
            if not os.path.exists(data_config_path):
                os.makedirs(data_config_path)
            data_config_path = self.output_config_path + "/" + self.data + ".yml"
            with open(data_config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)


if __name__ == "__main__":
    args = parser.parse_args()
    a = Dataconfig(args)
