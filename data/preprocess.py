from logging import raiseExceptions
import pandas as pd
import numpy as np


def generate_normalized_feature(df):
    df["zopen"] = df["open"] / df["close"] - 1
    df["zhigh"] = df["high"] / df["close"] - 1
    df["zlow"] = df["low"] / df["close"] - 1
    df["zadjcp"] = df["adjcp"] / df["close"] - 1
    df_new = df.sort_values(by=["tic", "date"])
    stock = df_new
    unique_ticker = df_new.tic.unique()
    df_indicator = pd.DataFrame()
    for i in range(len(unique_ticker)):
        temp_indicator = stock[stock.tic == unique_ticker[i]]
        temp_indicator["zclose"] = (
            temp_indicator.close /
            (temp_indicator.close.rolling(2).sum() - temp_indicator.close)) - 1
        temp_indicator["zd_5"] = (temp_indicator.adjcp.rolling(5).sum() /
                                  5) / temp_indicator.adjcp - 1
        temp_indicator["zd_10"] = (temp_indicator.adjcp.rolling(10).sum() /
                                   10) / temp_indicator.adjcp - 1
        temp_indicator["zd_15"] = (temp_indicator.adjcp.rolling(15).sum() /
                                   15) / temp_indicator.adjcp - 1
        temp_indicator["zd_20"] = (temp_indicator.adjcp.rolling(20).sum() /
                                   20) / temp_indicator.adjcp - 1
        temp_indicator["zd_25"] = (temp_indicator.adjcp.rolling(25).sum() /
                                   25) / temp_indicator.adjcp - 1
        temp_indicator["zd_30"] = (temp_indicator.adjcp.rolling(30).sum() /
                                   30) / temp_indicator.adjcp - 1
        df_indicator = df_indicator.append(temp_indicator, ignore_index=True)
    df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
    return df_indicator


def get_date(df):
    date = df.date.unique()
    start_date = pd.to_datetime(date[0])
    end_date = pd.to_datetime(date[-1])
    return start_date, end_date


def data_split(df, start, end, target_date_col="date"):

    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def split(data, portion):
    """ split the data by the portion into train, valid, test, which is convinent for the users to do the time rolling
    experiment"""
    portion = np.array(portion)
    if np.sum(portion) > 1:
        portion = normalization(portion)
    if (len(portion) in [3]) == False:
        raiseExceptions("we can only split data 3 parts: train, valid ,test")
    start, end = get_date(data)
    duration_length = end - start
    train_len = duration_length * portion[0]
    valid_len = duration_length * portion[1]
    test_len = duration_length * portion[2]
    train_end = start + train_len
    valid_start = train_end
    valid_end = valid_start + valid_len
    test_start = valid_end
    test_end = test_start + test_len
    start = str(start)
    train_end = str(train_end)
    valid_start = str(valid_start)
    valid_end = str(valid_end)
    test_start = str(test_start)
    test_end = str(test_end)
    train = data_split(data, start, train_end)
    valid = data_split(data, valid_start, valid_end)
    test = data_split(data, test_start, test_end)
    return train, valid, test


def normalization(portion):
    portion = np.array(portion)
    sum = np.sum(portion)
    portion = portion / sum
    return portion


if __name__ == "__main__":
    data = pd.read_csv("data/data/BTC_even/BTC.csv", index_col=0)
    train, valid, test = split(data, [1, 1, 1])
    train.to_csv("data/data/BTC_even/train.csv")
    valid.to_csv("data/data/BTC_even/valid.csv")
    test.to_csv("data/data/BTC_even/test.csv")
