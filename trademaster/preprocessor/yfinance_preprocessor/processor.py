from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import os.path as osp
from ..custom import CustomPreprocessor
from ..builder import PREPROCESSOR
from trademaster.utils import get_attr
import pandas as pd
import os
import yfinance as yf
from tqdm import tqdm
import numpy as np

@PREPROCESSOR.register_module()
class YfinancePreprocessor(CustomPreprocessor):
    def __init__(self, **kwargs):
        super(YfinancePreprocessor, self).__init__()
        self.kwargs = kwargs

        self.data_path = osp.join(ROOT, get_attr(kwargs, "data_path", None))
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.train_valid_test_portion = get_attr(kwargs,"train_valid_test_portion", [0.8, 0.1, 0.1])

        self.train_path = osp.join(ROOT, get_attr(kwargs, "train_path", None))
        self.valid_path = osp.join(ROOT, get_attr(kwargs, "valid_path", None))
        self.test_path = osp.join(ROOT, get_attr(kwargs, "test_path", None))

        self.start_date = get_attr(kwargs, "start_date", "2000-01-01")
        self.end_date = get_attr(kwargs, "end_date", "2019-01-01")
        self.tickers = get_attr(kwargs, "tickers", None)

    def download_data(self):
        df_list = []
        for ticker in self.tickers:
            df = yf.download(ticker, self.start_date, self.end_date)
            df["ticker"] = ticker
            df.index = df.index.values
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df = df.sort_values(by="ticker")
        df = df.sort_index()
        df["date"]=df.index
        return df

    def clean_data(self):
        initial_ticker_list = self.df[self.df.index == self.df.index.unique()
                                      [0]]["ticker"].values.tolist()
        initial_ticker_list = set(initial_ticker_list)
        for index in tqdm(self.df.index.unique()):
            ticker_list = self.df[self.df.index ==
                                  index]["ticker"].values.tolist()
            ticker_list = set(ticker_list)
            initial_ticker_list = initial_ticker_list & ticker_list
        df_list = []
        for ticker in initial_ticker_list:
            new_df = self.df[self.df.ticker == ticker]
            df_list.append(new_df)
        df = pd.concat(df_list)
        df = df.sort_values(by="ticker")
        df = df.sort_index()
        df = df.copy()
        df = df.sort_values(["date", "ticker"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date",
                                       columns="ticker",
                                       values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.ticker.isin(tics)]
        return df

    def make_feature(self):
        self.df["zopen"] = self.df["open"] / self.df["close"] - 1
        self.df["zhigh"] = self.df["high"] / self.df["close"] - 1
        self.df["zlow"] = self.df["low"] / self.df["close"] - 1
        self.df["zadjcp"] = self.df["adjclose"] / self.df["close"] - 1
        df_new = self.df.sort_values(by=["ticker", "date"])
        stock = df_new
        unique_ticker = stock.ticker.unique()
        df_indicator = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.ticker == unique_ticker[i]]
            temp_indicator["zclose"] = (temp_indicator.close /
                                        (temp_indicator.close.rolling(2).sum()
                                         - temp_indicator.close)) - 1
            temp_indicator["zd_5"] = (temp_indicator.close.rolling(5).sum() /
                                      5) / temp_indicator.close - 1
            temp_indicator["zd_10"] = (temp_indicator.close.rolling(10).sum() /
                                       10) / temp_indicator.close - 1
            temp_indicator["zd_15"] = (temp_indicator.close.rolling(15).sum() /
                                       15) / temp_indicator.close - 1
            temp_indicator["zd_20"] = (temp_indicator.close.rolling(20).sum() /
                                       20) / temp_indicator.close - 1
            temp_indicator["zd_25"] = (temp_indicator.close.rolling(25).sum() /
                                       25) / temp_indicator.close - 1
            temp_indicator["zd_30"] = (temp_indicator.close.rolling(30).sum() /
                                       30) / temp_indicator.close - 1
            df_indicator = df_indicator.append(temp_indicator,
                                               ignore_index=True)
        df_indicator = df_indicator.fillna(method="ffill").fillna(
            method="bfill")
        return df_indicator
    
    
    def get_date(self,df):
        date = df.date.unique()
        start_date = pd.to_datetime(date[0])
        end_date = pd.to_datetime(date[-1])
        return start_date, end_date


    def data_split(self,df, start, end, target_date_col="date"):

        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "ticker"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data


    def split(self,data, portion):
        """ split the data by the portion into train, valid, test, which is convinent for the users to do the time rolling
        experiment"""
        portion = np.array(portion)
        if np.sum(portion) > 1:
            portion = self.normalization(portion)
        if (len(portion) in [3]) == False:
            raise Exception("we can only split data 3 parts: train, valid ,test")
        start, end = self.get_date(data)
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
        train = self.data_split(data, start, train_end)
        valid = self.data_split(data, valid_start, valid_end)
        test = self.data_split(data, test_start, test_end)
        return train, valid, test


    def normalization(self,portion):
        portion = np.array(portion)
        sum = np.sum(portion)
        portion = portion / sum
        return portion

    def run(self, custom_data_path = None):

        if not custom_data_path:
            self.df = self.download_data()
        else:
            self.df = pd.read_csv(custom_data_path)

        self.df = self.df.rename(columns={
            "Open":"open",
            "High":"high",
            "Low":"low",
            "Close":"close",
            "Adj Close":"adjclose",
            "Volume":"volume",
            "ticker":"ticker",
            "date":"date"
        })
        self.df.index = self.df.date.values

        self.df = self.clean_data()

        self.df = self.make_feature()

        self.df = self.df.rename(columns={"ticker": "tic"})

        train, valid, test=self.split(self.df, self.train_valid_test_portion)

        train.to_csv(self.train_path)
        valid.to_csv(self.valid_path)
        test.to_csv(self.test_path)
