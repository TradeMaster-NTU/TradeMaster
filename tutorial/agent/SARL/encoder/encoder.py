import argparse
import yaml
import os
import torch
import numpy as np
import random
import sys

sys.path.append("./")

from agent.SARL.encoder.util import prepart_lstm_data, pick_optimizer, LDdataset, dict_to_args, prepart_m_lstm_data, m_lstm_dataset
from agent.SARL.model.net import LSTM_clf, m_LSTM_clf
from data.download_data import Dataconfig
from torch.utils.data import DataLoader
from agent.SARL.encoder.train_valid import train_with_valid

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
    "--encoder_path",
    type=str,
    default="./result/SARL/encoder",
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
    default="config/input_config/agent/SARL/encoder.yml",
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


class encoder:
    def __init__(self, args):
        if args.input_encoder_config == False:
            self.dataconfig = Dataconfig(args)
            self.encoder_path = args.encoder_path
            self.make_dict()
            self.seed = args.seed
            self.set_seed()
            self.batch_size = args.batch_size
            self.hidden_size = args.hidden_size
            self.num_epoch = args.num_epoch
            self.optimizer = pick_optimizer(args.optimizer)
            self.num_day = args.num_day
            self.technical_indicator = args.technical_indicator

        else:
            with open(args.input_encoder_config_dict, "r") as file:
                self.config = yaml.safe_load(file)
            args = dict_to_args(self.config)
            self.dataconfig = Dataconfig(args)
            self.encoder_path = self.config["encoder_path"]
            self.make_dict()
            self.seed = self.config["seed"]
            self.set_seed()
            self.batch_size = self.config["batch_size"]
            self.hidden_size = self.config["hidden_size"]
            self.num_epoch = self.config["num_epoch"]
            self.optimizer = pick_optimizer(self.config["optimizer"])
            self.num_day = self.config["num_day"]
            self.technical_indicator = args.technical_indicator

        self.train_data = self.dataconfig.train_dataset
        self.valid_data = self.dataconfig.valid_dataset
        self.test_data = self.dataconfig.test_dataset
        tic_list = self.train_data.tic.unique()

        self.train_label_list, self.train_df_list = prepart_m_lstm_data(
            self.train_data, self.num_day, self.technical_indicator)
        self.valid_label_list, self.valid_df_list = prepart_m_lstm_data(
            self.valid_data, self.num_day, self.technical_indicator)
        self.test_label_list, self.test_df_list = prepart_m_lstm_data(
            self.test_data, self.num_day, self.technical_indicator)
        self.train_dataset = m_lstm_dataset(self.train_df_list,
                                            self.train_label_list)
        self.valid_dataset = m_lstm_dataset(self.valid_df_list,
                                            self.valid_label_list)
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        valid_dataloader = DataLoader(self.valid_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.set_seed()
        self.net = m_LSTM_clf(self.train_df_list.shape[-1], args.num_layer,
                              self.hidden_size, len(tic_list))
        self.make_dict()
        self.optimizer_instance = self.optimizer(self.net.parameters(),
                                                 lr=args.lr)
        train_with_valid(train_dataloader, valid_dataloader, self.num_epoch,
                         self.net, self.optimizer_instance, self.encoder_path)

    def make_dict(self):
        if not os.path.exists(self.encoder_path):
            os.makedirs(self.encoder_path)

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.seed = self.seed


if __name__ == "__main__":
    args = parser.parse_args()

    a = encoder(args)