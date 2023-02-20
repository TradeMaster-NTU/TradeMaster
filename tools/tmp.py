import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

import pandas as pd

def process1():
    data_path = os.path.join(ROOT, "data", "algorithmic_trading","BTC")
    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "valid.csv")
    test_path = os.path.join(data_path, "test.csv")

    train_data = pd.read_csv(train_path, index_col=0)
    val_data = pd.read_csv(val_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    val_data.index = val_data.index + len(train_data)
    test_data.index = test_data.index + len(train_data) + len(val_data)

    print("index min = {}, index max = {}, date min = {}, date max = {}".format(train_data.index.min(),
                                                                                train_data.index.max(),
                                                                                train_data.date.min(),
                                                                                train_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(val_data.index.min(),
                                                                                val_data.index.max(),
                                                                                val_data.date.min(),
                                                                                val_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(test_data.index.min(),
                                                                                test_data.index.max(),
                                                                                test_data.date.min(),
                                                                                test_data.date.max()))

    data = pd.concat([train_data, val_data, test_data])
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(data.index.min(),
                                                                                data.index.max(),
                                                                                data.date.min(),
                                                                                data.date.max()))
    data.to_csv(os.path.join(data_path, "data.csv"))

def process2():
    data_path = os.path.join(ROOT, "data", "order_execution", "BTC")
    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "valid.csv")
    test_path = os.path.join(data_path, "test.csv")

    train_data = pd.read_csv(train_path, index_col=0)
    val_data = pd.read_csv(val_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    val_data.index = val_data.index + len(train_data)
    test_data.index = test_data.index + len(train_data) + len(val_data)

    print("index min = {}, index max = {}, date min = {}, date max = {}".format(train_data.index.min(),
                                                                                train_data.index.max(),
                                                                                train_data.system_time.min(),
                                                                                train_data.system_time.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(val_data.index.min(),
                                                                                val_data.index.max(),
                                                                                val_data.system_time.min(),
                                                                                val_data.system_time.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(test_data.index.min(),
                                                                                test_data.index.max(),
                                                                                test_data.system_time.min(),
                                                                                test_data.system_time.max()))

    data = pd.concat([train_data, val_data, test_data])
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(data.index.min(),
                                                                                data.index.max(),
                                                                                data.system_time.min(),
                                                                                data.system_time.max()))
    data = data.rename(columns={"system_time":"date"})
    data.to_csv(os.path.join(data_path, "data.csv"))

def process3():
    data_path = os.path.join(ROOT, "data", "order_execution", "PD_BTC")
    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "valid.csv")
    test_path = os.path.join(data_path, "test.csv")

    train_data = pd.read_csv(train_path, index_col=0)
    val_data = pd.read_csv(val_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    val_data.index = val_data.index + len(train_data)
    test_data.index = test_data.index + len(train_data) + len(val_data)

    print("index min = {}, index max = {}, date min = {}, date max = {}".format(train_data.index.min(),
                                                                                train_data.index.max(),
                                                                                train_data.date.min(),
                                                                                train_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(val_data.index.min(),
                                                                                val_data.index.max(),
                                                                                val_data.date.min(),
                                                                                val_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(test_data.index.min(),
                                                                                test_data.index.max(),
                                                                                test_data.date.min(),
                                                                                test_data.date.max()))

    data = pd.concat([train_data, val_data, test_data])
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(data.index.min(),
                                                                                data.index.max(),
                                                                                data.date.min(),
                                                                                data.date.max()))
    data.to_csv(os.path.join(data_path, "data.csv"))

def process4():
    data_path = os.path.join(ROOT, "data", "portfolio_management", "dj30")
    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "valid.csv")
    test_path = os.path.join(data_path, "test.csv")

    train_data = pd.read_csv(train_path, index_col=0)
    val_data = pd.read_csv(val_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    val_data.index = val_data.index + train_data.index.max() + 1
    test_data.index = test_data.index + val_data.index.max() + 1

    print("index min = {}, index max = {}, date min = {}, date max = {}".format(train_data.index.min(),
                                                                                train_data.index.max(),
                                                                                train_data.date.min(),
                                                                                train_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(val_data.index.min(),
                                                                                val_data.index.max(),
                                                                                val_data.date.min(),
                                                                                val_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(test_data.index.min(),
                                                                                test_data.index.max(),
                                                                                test_data.date.min(),
                                                                                test_data.date.max()))

    data = pd.concat([train_data, val_data, test_data])
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(data.index.min(),
                                                                                data.index.max(),
                                                                                data.date.min(),
                                                                                data.date.max()))
    data.to_csv(os.path.join(data_path, "data.csv"))

def process5():
    data_path = os.path.join(ROOT, "data", "portfolio_management", "exchange")
    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "valid.csv")
    test_path = os.path.join(data_path, "test.csv")

    train_data = pd.read_csv(train_path, index_col=0)
    val_data = pd.read_csv(val_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    val_data.index = val_data.index + train_data.index.max() + 1
    test_data.index = test_data.index + val_data.index.max() + 1

    print("index min = {}, index max = {}, date min = {}, date max = {}".format(train_data.index.min(),
                                                                                train_data.index.max(),
                                                                                train_data.date.min(),
                                                                                train_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(val_data.index.min(),
                                                                                val_data.index.max(),
                                                                                val_data.date.min(),
                                                                                val_data.date.max()))
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(test_data.index.min(),
                                                                                test_data.index.max(),
                                                                                test_data.date.min(),
                                                                                test_data.date.max()))

    data = pd.concat([train_data, val_data, test_data])
    print("index min = {}, index max = {}, date min = {}, date max = {}".format(data.index.min(),
                                                                                data.index.max(),
                                                                                data.date.min(),
                                                                                data.date.max()))
    data.to_csv(os.path.join(data_path, "data.csv"))

if __name__=="__main__":
    # process1()
    # process2()
    # process3()
    # process4()
    # process5()
    import numpy as np

    data = np.load("industry_classification.npy")
    print(data.shape)
    print(data)