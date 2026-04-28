import multiprocessing
from copy import deepcopy
from datetime import datetime
from finagent.registry import PROCESSOR
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from langchain_community.document_loaders import PlaywrightURLLoader
import backoff
import time

@backoff.on_exception(backoff.expo,(Exception,), max_tries=3, max_value=10, jitter=None)
def langchain_parse_url(url):

    print(">" * 30 + "Running langchain_parse_url" + ">" * 30)
    start = time.time()
    loader = PlaywrightURLLoader(urls = [url], remove_selectors=["header", "footer"])
    data = loader.load()
    if len(data) <= 0:
        return None
    content = data[0].page_content

    if "Please enable cookies" in content:
        return None
    if "Please verify you are a human" in content:
        return None
    if "Checking if the site connection is secure" in content:
        return None

    print("url: {} | content: {}".format(url, content[:100].split("\n")))
    end = time.time()
    print(">" * 30 + "Time elapsed: {}s".format(end - start) + ">" * 30)
    print("<" * 30 + "Finish langchain_parse_url" + "<" * 30)
    return content

def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

def cal_news(df):
    df["title"] = df["title"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["text"] = df["text"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["source"] = df["source"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return df

def cal_guidance(df):
    df["title"] = df["title"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["text"] = df["text"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return df

def cal_sentiment(df, columns):
    for col in columns:
        if "sentiment" not in col:
            df[col] = df.groupby("timestamp")[col].transform("sum")
        else:
            df[col] = df.groupby("timestamp")[col].transform("mean")
            df[col] = df[col].fillna(0.5)
    return df


ADANOS_SENTIMENT_COLUMNS = [
    "adanos_source",
    "adanos_buzz_score",
    "adanos_mentions",
    "adanos_sentiment_score",
    "adanos_bullish_pct",
    "adanos_bearish_pct",
    "adanos_trend",
    "adanos_latest_buzz_score",
    "adanos_latest_sentiment_score",
]


def cal_factor(df, level="day"):
    # intermediate values
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    df["kmid"] = (df["close"] - df["open"]) / df["close"]
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df["klen"] = (df["high"] - df["low"]) / df["open"]
    df['kup'] = (df['high'] - df['max_oc']) / df['open']
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df["ksft"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    window = [5, 10, 20, 30, 60]
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    for w in window:
        df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    for w in window:
        shift = df['close'].shift(w)
        min = df["low"].where(df["low"] < shift, shift)
        max = df["high"].where(df["high"] > shift, shift)
        df["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    for w in window:
        df1 = df["close"].rolling(w)
        df2 = np.log(df["volume"] + 1).rolling(w)
        df["corr_{}".format(w)] = df1.corr(pairwise = df2)

    for w in window:
        df1 = df["close"]
        df_shift1 = df1.shift(1)
        df2 = df["volume"]
        df_shift2 = df2.shift(1)
        df1 = df1 / df_shift1
        df2 = np.log(df2 / df_shift2 + 1)
        df["cord_{}".format(w)] = df1.rolling(w).corr(pairwise = df2.rolling(w))

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    for w in window:
        df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    for w in window:
        df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    for w in window:
        df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1

    for w in window:
        df["vma_{}".format(w)] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-12)

    for w in window:
        df["vstd_{}".format(w)] = df["volume"].rolling(w).std() / (df["volume"] + 1e-12)

    for w in window:
        shift = np.abs((df["close"] / df["close"].shift(1) - 1)) * df["volume"]
        df1 = shift.rolling(w).std()
        df2 = shift.rolling(w).mean()
        df["wvma_{}".format(w)] = df1 / (df2 + 1e-12)

    df['vchg1'] = df['volume'] - df['volume'].shift(1)
    df['abs_vchg1'] = np.abs(df['vchg1'])
    df['pos_vchg1'] = df['vchg1']
    df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    for w in window:
        df["vsump_{}".format(w)] = df["pos_vchg1"].rolling(w).sum() / (df["abs_vchg1"].rolling(w).sum() + 1e-12)
    for w in window:
        df["vsumn_{}".format(w)] = 1 - df["vsump_{}".format(w)]
    for w in window:
        df["vsumd_{}".format(w)] = 2 * df["vsump_{}".format(w)] - 1

    df["log_volume"] = np.log(df["volume"] + 1)

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1', 'volume'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)

    if level == "minute":
        df["minute"] = pd.to_datetime(df.index).minute
        df["hour"] = pd.to_datetime(df.index).hour

    df["day"] = pd.to_datetime(df.index).day
    df["weekday"] = pd.to_datetime(df.index).weekday
    df["month"] = pd.to_datetime(df.index).month

    return df

def cal_target(df):
    df['ret1'] = df['close'].pct_change(1).shift(-1)
    df['mov1'] = (df['ret1'] > 0)
    df['mov1'] = df['mov1'].astype(int)
    return df

@PROCESSOR.register_module(force=True)
class Processor():
    def __init__(self,
                 root=None,
                 path_params = None,
                 stocks_path = None,
                 start_date = None,
                 end_date = None,
                 interval="day",
                 if_parse_url = False,
                 workdir = None,
                 tag = None
                 ):
        self.root = root
        self.path_params = path_params
        self.stocks_path = os.path.join(root, stocks_path)
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.if_parse_url = if_parse_url
        self.workdir = workdir
        self.tag = tag

        self.stocks = self._init_stocks()

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()]
        return stocks

    def _process_price_and_features(self,
                stocks = None,
                start_date = None,
                end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close"
        ]

        for stock in tqdm(stocks):
            price = self.path_params["prices"][0]
            price_type = price["type"]
            price_path = price["path"]

            price_path = os.path.join(self.root, price_path, "{}.csv".format(stock))

            if price_type == "fmp":
                price_column_map = {
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "adjClose": "adj_close",
                }
            elif price_type == "yahoofinance":
                price_column_map = {
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Date": "timestamp",
                    "Adj Close": "adj_close",
                }
            else:
                price_column_map = {
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "adjClose": "adj_close",
                }

            assert os.path.exists(price_path), "Price path {} does not exist".format(price_path)
            price_df = pd.read_csv(price_path)

            price_df = price_df.rename(columns=price_column_map)[["timestamp"] + price_columns]

            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
            price_df = price_df[(price_df["timestamp"] >= start_date) & (price_df["timestamp"] < end_date)]

            price_df = price_df.sort_values(by="timestamp")
            price_df = price_df.drop_duplicates(subset=["timestamp"], keep="first")
            price_df = price_df.reset_index(drop=True)

            outpath = os.path.join(self.root, self.workdir, self.tag, "price")
            os.makedirs(outpath, exist_ok=True)
            price_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)

            features_df = cal_factor(deepcopy(price_df), level=self.interval)
            features_df = cal_target(features_df)
            outpath = os.path.join(self.root, self.workdir, self.tag, "features")
            os.makedirs(outpath, exist_ok=True)
            features_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)

    def _process_guidance(self,
                          stocks = None,
                          start_date = None,
                          end_date = None):
        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        guidance_columns = [
            "title",
            "text",
            "sentiment",
            "url",
        ]
        
        for stock in tqdm(stocks):

            guidances = self.path_params["guidance"]
            guidances_df = []

            for guidance in guidances:
                guidance_type = guidance["type"]
                guidance_path = guidance["path"]

                guidance_path = os.path.join(self.root, guidance_path, "{}.csv".format(stock))

                if guidance_type == "rapidapi_seekingalpha":
                    guidance_column_map = {
                        "title": "title",
                        "summary": "text",
                        "sentiment": "sentiment",
                        "url": "url",
                    }

                assert os.path.exists(guidance_path), "guidance path {} does not exist".format(guidance_path)

                guidance_df = pd.read_csv(guidance_path)
                guidance_df = guidance_df.rename(columns=guidance_column_map)[["timestamp"] + guidance_columns]
                guidance_df["timestamp"] = pd.to_datetime(guidance_df["timestamp"])

                guidance_df = guidance_df[(guidance_df["timestamp"] >= start_date) & (guidance_df["timestamp"] < end_date)]
                guidance_df = guidance_df.sort_values(by="timestamp")
                guidance_df = guidance_df.drop_duplicates(subset=["timestamp", "title", "text"], keep="first")

                if guidance_type == "rapidapi_seekingalpha":
                    guidance_df["type"] = "rapidapi"
                    guidance_df["source"] = "seekingalpha"

                guidance_df = guidance_df.reset_index(drop=True)
                guidance_df = cal_guidance(guidance_df)
                guidance_df["timestamp"] = pd.to_datetime(guidance_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d"))
                guidances_df.append(guidance_df)

            guidances_df = pd.concat(guidances_df)

            if self.if_parse_url:
                urls = guidances_df["url"].values
                max_process = 10
                pool = multiprocessing.Pool(processes=max_process)
                contents = pool.map(langchain_parse_url, urls)
                pool.close()
                pool.join()
                guidances_df["content"] = contents

            guidances_df = guidances_df.sort_values(by="timestamp")
            guidances_df = guidances_df.reset_index(drop=True)
            guidances_df = guidances_df[["timestamp", "type", "sentiment", "title", "text", "url"]]

            outpath = os.path.join(self.root, self.workdir, self.tag, "guidance")
            os.makedirs(outpath, exist_ok=True)
            guidances_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)

    def _process_news(self,
                stocks = None,
                start_date = None,
                end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        news_columns = [
            "title",
            "text",
            "source",
            "url"
        ]

        for stock in tqdm(stocks):

            newses = self.path_params["news"]
            newses_df = []

            for news in newses:
                news_type = news["type"]
                news_path = news["path"]

                news_path = os.path.join(self.root, news_path, "{}.csv".format(stock))

                if news_type == "fmp":
                    news_column_map = {
                        "title": "title",
                        "text": "text",
                        "site": "source",
                        "url": "url",
                    }
                elif news_type == "yahoofinance":
                    news_column_map = {
                        "headline": "title",
                        "summary": "text",
                        "datetime": "timestamp",
                        "source": "source",
                        "url": "url",
                    }
                else:
                    news_column_map = {
                        "title": "title",
                        "text": "text",
                        "site": "source",
                        "url": "url",
                    }

                assert os.path.exists(news_path), "News path {} does not exist".format(news_path)

                news_df = pd.read_csv(news_path)
                news_df = news_df.rename(columns=news_column_map)[["timestamp"] + news_columns]
                news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])

                news_df = news_df[(news_df["timestamp"] >= start_date) & (news_df["timestamp"] < end_date)]
                news_df = news_df.sort_values(by="timestamp")
                news_df = news_df.drop_duplicates(subset=["timestamp", "title", "text"], keep="first")

                if news_type == "fmp":
                    news_df["type"] = "fmp"
                elif news_type == "yahoofinance":
                    news_df["type"] = "yahoofinance"

                news_df = news_df.reset_index(drop=True)
                news_df = cal_news(news_df)
                news_df["timestamp"] = pd.to_datetime(news_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d"))
                newses_df.append(news_df)

            newses_df = pd.concat(newses_df)

            if self.if_parse_url:
                urls = newses_df["url"].values
                max_process = 10
                pool = multiprocessing.Pool(processes=max_process)
                contents = pool.map(langchain_parse_url, urls)
                pool.close()
                pool.join()
                newses_df["content"] = contents

            newses_df = newses_df.sort_values(by="timestamp")
            newses_df = newses_df.drop_duplicates(subset=["timestamp", "title"], keep="first")
            newses_df = newses_df.reset_index(drop=True)
            newses_df = newses_df[["timestamp", "type", "source", "title", "text", "url"]]

            outpath = os.path.join(self.root, self.workdir, self.tag, "news")
            os.makedirs(outpath, exist_ok=True)
            newses_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)

    def _process_sentiment(self,
                           stocks = None,
                           start_date = None,
                           end_date = None):
        """
        stocktwits_posts,twitter_posts,
        stocktwits_comments,twitter_comments,
        stocktwits_likes,twitter_likes,
        stocktwits_impressions,twitter_impressions,
        stocktwits_sentiment,twitter_sentiment
        """

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        sentiment_columns = [
            "stocktwits_posts",
            "stocktwits_comments",
            "stocktwits_likes",
            "stocktwits_impressions",
            "stocktwits_sentiment",
        ]

        for stock in tqdm(stocks):

            sentiments = self.path_params["sentiment"]
            sentiments_df = []

            for sentiment in sentiments:
                sentiment_type = sentiment["type"]
                sentiment_path = sentiment["path"]

                sentiment_path = os.path.join(self.root, sentiment_path, "{}.csv".format(stock))

                if sentiment_type == "fmp":
                    sentiment_column_map = {}
                    selected_columns = sentiment_columns
                elif sentiment_type == "adanos":
                    sentiment_column_map = {}
                    selected_columns = ADANOS_SENTIMENT_COLUMNS
                else:
                    raise ValueError("Unsupported sentiment_type: {}".format(sentiment_type))

                assert os.path.exists(sentiment_path), "sentiment path {} does not exist".format(sentiment_path)

                sentiment_df = pd.read_csv(sentiment_path)
                sentiment_df = sentiment_df.rename(columns=sentiment_column_map)[["timestamp"] + selected_columns]
                sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"])

                sentiment_df = sentiment_df[ (sentiment_df["timestamp"] >= start_date) & (sentiment_df["timestamp"] < end_date)]
                sentiment_df = sentiment_df.sort_values(by="timestamp")
                sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d"))

                if sentiment_type == "adanos":
                    sentiment_df = sentiment_df.drop_duplicates(subset=["timestamp", "adanos_source"], keep="last")
                else:
                    sentiment_df = cal_sentiment(sentiment_df, sentiment_columns)
                    sentiment_df = sentiment_df.drop_duplicates(subset=["timestamp"], keep="first")
                sentiment_df = sentiment_df.reset_index(drop=True)
                sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d"))
                sentiments_df.append(sentiment_df)

            sentiments_df = pd.concat(sentiments_df)

            if self.if_parse_url:
                urls = sentiments_df["url"].values
                max_process = 10
                pool = multiprocessing.Pool(processes=max_process)
                contents = pool.map(langchain_parse_url, urls)
                pool.close()
                pool.join()
                sentiments_df["content"] = contents

            sentiments_df["type"] = "sentiment"

            sentiments_df = sentiments_df.sort_values(by="timestamp")
            sentiments_df = sentiments_df.reset_index(drop=True)
            if "adanos_source" in sentiments_df.columns:
                sentiments_df = sentiments_df[["timestamp", "type"] + ADANOS_SENTIMENT_COLUMNS]
            else:
                sentiments_df = sentiments_df[["timestamp", "type"] + sentiment_columns]

            outpath = os.path.join(self.root, self.workdir, self.tag, "sentiment")
            os.makedirs(outpath, exist_ok=True)
            sentiments_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)

    def _process_economic(self,
                          stocks = None,
                          start_date = None,
                          end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        """
        GDP, realGDP, nominalPotentialGDP, realGDPPerCapita, federalFunds, CPI, inflationRate, inflation, retailSales, consumerSentiment, durableGoods, unemploymentRate, totalNonfarmPayroll, initialClaims, industrialProductionTotalIndex, newPrivatelyOwnedHousingUnitsStartedTotalUnits, totalVehicleSales, retailMoneyFunds, smoothedUSRecessionProbabilities, 3MonthOr90DayRatesAndYieldsCertificatesOfDeposit, commercialBankInterestRateOnCreditCardPlansAllAccounts, 30YearFixedRateMortgageAverage, 15YearFixedRateMortgageAverage
        """

        indicators = [
            "GDP",
            "federalFunds",
            "CPI",
            "inflationRate",
            "unemploymentRate",
        ]

        type = self.path_params["economic"][0]["type"]
        path = self.path_params["economic"][0]["path"]

        df = None

        for indicator in indicators:

            indicator_path = os.path.join(self.root, path, "{}.csv".format(indicator))
            assert os.path.exists(indicator_path), "indicator path {} does not exist".format(indicator_path)

            indicator_df = pd.read_csv(indicator_path)

            indicator_df = indicator_df.rename(columns={
                "GDP": "gdp",
                "federalFunds": "federal_funds",
                "CPI": "cpi",
                "inflationRate": "inflation_rate",
                "unemploymentRate": "unemployment_rate",
            })

            indicator_df["timestamp"] = pd.to_datetime(indicator_df["timestamp"])
            indicator_df = indicator_df[(indicator_df["timestamp"] >= start_date) & (indicator_df["timestamp"] < end_date)]
            indicator_df = indicator_df.sort_values(by="timestamp")

            if df is None:
                df = indicator_df
            else:
                df = pd.merge(df, indicator_df, on="timestamp", how="left")

            df = df.fillna(method="ffill")
            df = df.fillna(method="bfill")
            df = df.reset_index(drop=True)
            df["type"] = "economic"

        df.to_parquet(os.path.join(self.root, self.workdir, self.tag, "economic.parquet"), index=False)

    def process(self,
                stocks = None,
                start_date = None,
                end_date = None):

        print(">" * 30 + "Running price and features..." + ">" * 30)
        self._process_price_and_features(stocks=stocks, start_date=start_date, end_date=end_date)
        print("<" * 30 + "Finish price and features..." + "<" * 30)

        if "guidance" in self.path_params:
            print(">" * 30 + "Running guidance..." + ">" * 30)
            self._process_guidance(stocks=stocks, start_date=start_date, end_date=end_date)
            print("<" * 30 + "Finish guidance..." + "<" * 30)

        if "sentiment" in self.path_params:
            print(">" * 30 + "Running sentiment..." + ">" * 30)
            self._process_sentiment(stocks=stocks, start_date=start_date, end_date=end_date)
            print("<" * 30 + "Finish sentiment..." + "<" * 30)

        print(">" * 30 + "Running news..." + ">" * 30)
        self._process_news(stocks=stocks, start_date=start_date, end_date=end_date)
        print("<" * 30 + "Finish news..." + "<" * 30)

        if "economic" in self.path_params:
            print(">" * 30 + "Running economic..." + ">" * 30)
            self._process_economic(stocks=stocks, start_date=start_date, end_date=end_date)
            print("<" * 30 + "Finish economic..." + "<" * 30)
