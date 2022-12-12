import pandas as pd
import numpy as np


def calculate_SMA(df: pd.DataFrame, num_day, feature="close"):
    SMA = df[feature].rolling(num_day, min_periods=0).sum() / num_day
    df["SMA_{}_{}".format(num_day, feature)] = SMA
    return df


def calculate_EMA(df: pd.DataFrame, num_day=10, feature="close"):
    # we use the formula EMA = (Closing Price - EMA(previous day)) x Weighting Multiplier + EMA(previous day)
    # to calculate the EMA and use the SMA as the first one
    EMA = df[feature].ewm(span=num_day, adjust=False).mean()
    df["EMA_{}_{}".format(num_day, feature)] = EMA
    return df


def calculate_RSI(df: pd.DataFrame, num_day=14, feature="close"):
    price_changes = df[feature].diff().fillna(0)
    gain = price_changes.copy()
    gain[gain < 0] = 0
    loss = price_changes.copy()
    loss[loss > 0] = 0
    rsi = 100-100/(1+gain.rolling(num_day, min_periods=0).mean()/-
                   loss.rolling(num_day, min_periods=0).mean())
    df["RSI_{}_{}".format(num_day, feature)] = rsi
    print(df)
    return df


def calculate_return(df: pd.DataFrame, feature="close"):
    df["return"] = df[feature].diff()
    df=df.fillna(0)
    return df


def calculate_MACD(df: pd.DataFrame):
    df["DIF"] = df["EMA_12_close"]-df["EMA_26_close"]
    df["MACD"] = df["DIF"].ewm(span=10, adjust=False).mean()
    return df


def caculate_BB(df: pd.DataFrame, feature="close"):
    df["BB_mid"] = df[feature].rolling(20, min_periods=0).sum() / 20
    df["BB_low"] = df[feature].rolling(20, min_periods=0).sum(
    ) / 20-2*df[feature].rolling(20, min_periods=0).std()
    df["BB_high"] = df[feature].rolling(20, min_periods=0).sum(
    ) / 20+2*df[feature].rolling(20, min_periods=0).std()
    return df


def calculate_slowK(df: pd.DataFrame):
    df["slowK"] = (df["close"]-df["low"])/(df["high"]-df["low"])
    return df


def preprocess(df: pd.DataFrame):
    df = calculate_EMA(df)
    df = calculate_EMA(df, 12)
    df = calculate_EMA(df, 26)
    df = calculate_RSI(df)
    df = calculate_return(df)
    df = calculate_MACD(df)
    df = caculate_BB(df)
    df = calculate_slowK(df)
    print(df.columns)
    df["best_action"]=(df["return"].values[1:]/np.abs(df["return"].values[1:])).tolist()+[0]
    price_feature=['open', 'close', 'high', 'low','EMA_12_close','EMA_26_close','EMA_10_close','BB_mid','BB_low','BB_high']
    precentile_feature=['MACD']
    for feature in price_feature:
        df["normal_{}".format(feature)]=df[feature]/10000# this is determined by the price of bitcoin
    for feature in precentile_feature:
        df["normal_{}".format(feature)]=df[feature]/100

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/data/BTC/test.csv", index_col=0)
    df=preprocess(df)
    df.to_csv("data/data/BTC_for_iRDPG/test.csv")
    
