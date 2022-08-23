from logging import raiseExceptions
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import argparse


def make_label_tic(num_day, df):
    max_index = df.index.max()
    # the target is to return number of tic list
    # for each element, it conisits of 4 things: df AR SR MDD
    df_new = df.sort_values(by=["tic", "date"])
    # defining a function to find out the sr mdd AR trading only one stock during a specific period
    tic_list = df.tic.unique()
    # here we first compute the  AR because for df, the interval is num_day
    AR_all = []
    df_all = []
    MDD_all = []
    SR_all = []
    ER_all = []
    WR_all = []
    for tic in tic_list:
        ARs = []
        MDDs = []
        SRs = []
        ERs = []
        WRs = []
        df_tic = df_new[df_new.tic == tic]
        for index in df_tic.index.unique()[:-num_day]:
            price_old = df_tic[df_tic.index == index]
            price_new = df_tic[df_tic.index == index + num_day]
            return_rate = (float(price_new.close) / float(price_old.close)) - 1
            AR = return_rate / num_day * 365
            ARs.append(AR)
            daily_return_rate = []
            average_return_rate = []
            for dex in range(index, index + num_day):
                old_price = df_tic[df_tic.index == dex]
                new_price = df_tic[df_tic.index == dex + 1]
                return_rate = (float(new_price.close) /
                               float(old_price.close)) - 1

                daily_return_rate.append(return_rate)
                old_acerage_price = np.average(df[df.index == dex].close)
                new_acerage_price = np.average(df[df.index == dex + 1].close)
                average_return_rate.append(new_acerage_price /
                                           old_acerage_price - 1)

            sr, mdd = evaluate(daily_return_rate)
            er, wr = compare_average(daily_return_rate, average_return_rate)
            MDDs.append(mdd)
            SRs.append(sr)
            ERs.append(er)
            WRs.append(wr)
        AR_all.append(ARs)
        df_all.append(df_tic[df_tic.index <= max_index - num_day].drop(
            ['tic', 'date'], axis=1))
        MDD_all.append(MDDs)
        SR_all.append(SRs)
        WR_all.append(WRs)
        ER_all.append(ERs)
    return df_all, AR_all, MDD_all, SR_all, ER_all, WR_all


def evaluate(return_list):
    length = len(return_list)
    mean = np.mean(return_list)
    std = np.std(return_list)
    sr = mean / std * 1 / np.sqrt(length)
    q = []
    for return_rate in return_list:
        c = return_rate + 1
        q.append(c)
    price_list = [1]
    for m in q:
        new = m * price_list[-1]
        price_list.append(new)
    DD = []
    for b in price_list:  # 选起点
        DD_numbers = []
        number = price_list.index(b)
        for c in range(number, length + 1):
            DD_numbers.append((b - price_list[c]) / b)
        DD_number = np.max(DD_numbers)
        DD.append(DD_number)
    MDD = np.max(DD)

    return sr, MDD


def compare_average(tic_return_list, average_return_list):
    if len(tic_return_list) != len(average_return_list):
        raiseExceptions("the 2 list should have the same length")
    length = len(tic_return_list)
    average_reward = []
    for average_return in average_return_list:
        average_reward.append(1 + average_return)
    average_price_list = [1]
    for m in average_reward:
        new = m * average_price_list[-1]
        average_price_list.append(new)
    tic_reward = []
    for tic_return in tic_return_list:
        tic_reward.append(1 + tic_return)
    tic_price_list = [1]
    for m in tic_reward:
        new = m * tic_price_list[-1]
        tic_price_list.append(new)
    ER = ((tic_price_list[-1]) / (average_price_list[-1]) - 1) * 365 / length
    winning_day = 0
    for i in range(length):
        if tic_return_list[i] >= average_return_list[i]:
            winning_day = winning_day + 1
    WR = winning_day / length

    return ER, WR


def make_rank_label(value_label, reverse=False):
    a = value_label.copy()
    l, w = value_label.shape
    for i in range(w):
        information = value_label[:, i].copy()
        information = list(information)
        information.sort(reverse=reverse)
        information = list(information)
        rank = []
        for c in value_label[:, i]:
            if np.isnan(c) == True:
                if reverse == False:
                    rank.append(len(information) - 1)
                else:
                    rank.append(0)
            else:
                rank.append(information.index(c))
        rank = np.array(rank)
        a[:, i] = rank
    return a


class LDdataset(Dataset):
    def __init__(self, df_list, rank):
        self.df = df_list
        self.label = rank
        for i in range(len(self.df)):
            if i == 0:
                data = torch.from_numpy(self.df[i].values).unsqueeze(0)
            else:
                data_new = torch.from_numpy(self.df[i].values).unsqueeze(0)
                data = torch.cat([data, data_new], dim=0)
        self.X = data
        self.y = torch.from_numpy(self.label)

    def __len__(self):
        return (len(self.df[0]))

    def __getitem__(self, idx):
        X = self.X[:, idx, :]
        y = self.y[:, idx]
        return X, y


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


def dict_to_args(**kwargs):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args()
    return args
