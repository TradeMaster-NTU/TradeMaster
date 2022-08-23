from logging import raiseExceptions
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import argparse


def prepart_m_lstm_data(df, num_day, technical_indicator):
    tic_list = df.tic.unique()
    df_list = []
    label_list = []
    for index in df.index.unique()[num_day:]:
        dfs = []
        labels = []
        df_date = df[[
            True if i in range(index - num_day, index) else False
            for i in df.index
        ]]
        for tic in tic_list:
            df_tic = df_date[df_date.tic == tic]
            np_tic = df_tic[technical_indicator].to_numpy()
            # print(np_tic.shape)
            dfs.append(np_tic)
            old_price = float(df_tic[df_tic.index == index - 1].close)
            new_price = float(df[(df.index == index) * (df.tic == tic)].close)
            if new_price > old_price:
                label = 1
            else:
                label = 0
            labels.append(label)
        df_list.append(dfs)
        label_list.append(labels)
    label_list = np.array(label_list)
    df_list = np.array(df_list)
    return label_list, df_list


def prepart_lstm_data(df, num_day, technical_indicator):
    # define a function to prepare X and y for lstm data
    tic_list = df.tic.unique()
    df_list = []
    label_list = []
    for tic in tic_list:
        dfs = []
        labels = []
        df_tic = df[df.tic == tic]
        for index in df_tic.index.unique()[num_day:]:
            df_information = df_tic[index - num_day:index][
                technical_indicator].to_numpy()
            dfs.append(df_information)
            old_price = float(df_tic[df_tic.index == index - 1].close)
            new_price = float(df_tic[df_tic.index == index].close)
            if new_price > old_price:
                label = 1
            else:
                label = 0
            labels.append(label)
        label_list.append(labels)
        df_list.append(dfs)
    label_list = np.array(label_list).reshape(-1)
    df_list = np.array(df_list).reshape(-1, num_day, len(technical_indicator))
    return label_list, df_list


def pick_optimizer(name_of_optimizer):
    if name_of_optimizer not in [
            "Adam", "SGD", "ASGD", "Rprop", "Adagrad", "Adadelta", "RMSprop",
            "Adamax", "SparseAdam", "LBFGS"
    ]:
        raiseExceptions(
            "Sorry, we still do not support this kind of optimizer")
    elif name_of_optimizer == "Adam":
        from torch.optim import Adam as optimizer
    elif name_of_optimizer == "SGD":
        from torch.optim import SGD as optimizer
    elif name_of_optimizer == "ASGD":
        from torch.optim import ASGD as optimizer
    elif name_of_optimizer == "Rprop":
        from torch.optim import Rprop as optimizer
    elif name_of_optimizer == "Adagrad":
        from torch.optim import Adagrad as optimizer
    elif name_of_optimizer == "Adadelta":
        from torch.optim import Adadelta as optimizer
    elif name_of_optimizer == "RMSprop":
        from torch.optim import RMSprop as optimizer
    elif name_of_optimizer == "Adamax":
        from torch.optim import Adamax as optimizer
    elif name_of_optimizer == "SparseAdam":
        from torch.optim import SparseAdam as optimizer
    elif name_of_optimizer == "LBFGS":
        from torch.optim import LBFGS as optimizer

    return optimizer


class LDdataset(Dataset):
    def __init__(self, df_list, label_list):
        self.df = df_list
        self.label = label_list
        self.X = torch.from_numpy(self.df).float()
        self.y = torch.from_numpy(self.label)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]
        return X, y


def dict_to_args(**kwargs):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args()
    return args


class m_lstm_dataset(Dataset):
    def __init__(self, df_list, label_list):
        self.df = df_list
        self.label = label_list
        self.X = torch.from_numpy(self.df).float()
        self.y = torch.from_numpy(self.label).float()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :, :]
        y = self.y[idx, :]
        return X, y
