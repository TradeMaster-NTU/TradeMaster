# Publication
[Deep Reinforcement Learning for Quantitative Trading: Challenges and Opportunities](https://ieeexplore.ieee.org/abstract/document/9779600) *(IEEE Intelligent Systems 2022)*

[DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities](https://arxiv.org/abs/2201.09058) *(CIKM 2022)*

[Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio Management](https://ojs.aaai.org/index.php/AAAI/article/view/16142) *(AAAI 21)*

[Reinforcement Learning for Quantitative Trading (Survey)](https://arxiv.org/abs/2109.13851)

[PRUDEX-Compass: Towards Systematic Evaluation of Reinforcement Learning in Financial Markets](https://openreview.net/forum?id=Vhb-awTdHCh)

# File Structure
```
|-- agent
|   |-- ClassicRL
|   |-- DeepScalper
|   |-- DeepTrader
|   |-- EIIE
|   |-- Investor_Imitator
|   |-- SARL
|-- config
|   |-- input_config
|   |-- output_config
|-- data
|   |-- download_data.py
|   |-- preprocess.py
|   |-- data
|       |-- BTC
|       |-- dj30
|       |-- exchange
|       |-- sz50
|-- env
|   |-- AT
|   |-- OE
|   |-- PM
|-- experiment
|-- figure
|-- result
|-- tutorial
|   |-- ClassRL_for_PM_on_FX.ipynb
|   |-- DeepScalper_for_AT_on_Bitcoin.ipynb
|   |-- EIIE_for_PM_on_DJ30.ipynb
|   |-- IMIT_for_PM_on_DJ30.ipynb
|   |-- SARL_for_PM_on_DJ30.ipynb
|   |-- Visualization.ipynb
|-- visualization
|   |-- compass
|   |-- exen
|   |-- ocatgon
|   |-- performance_profile
|   |-- rank
|-- README.md
|-- requirement.txt

```

# External Sources

Users may download data from the following data source with personal account:
| Data Source                                                                                   | Type                   | Range and Frequency      | Request Limits        | Raw Data              |
| --------------------------------------------------------------------------------------------- | ---------------------- | ------------------------ | --------------------- | --------------------- |
| [Alpaca](https://alpaca.markets/docs/introduction/)                                           | US Stocks, ETFs        | 2015-now, 1min           | Account-specific      | OHLCV                 |
| [Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)               | CN Securities          | 1990-12-19-now, 5min     | Account-specific      | OHLCV                 |
| [Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)             | Cryptocurrency         | API-specific, 1s, 1min   | API-specific          | Tick-level daily data |
| [CCXT](https://docs.ccxt.com/en/latest/manual.html)                                           | Cryptocurrency         | API-specific, 1min       | API-specific          | OHLCV                 |
| [IEXCloud](https://iexcloud.io/docs/api/)                                                     | NMS US securities      | 1970-now, 1 day          | 100 per second per IP | OHLCV                 |
| [JoinQuant](https://www.joinquant.com/)                                                       | CN Securities          | 2005-now, 1min           | 3 requests each time  | OHLCV                 |
| [QuantConnect](https://www.quantconnect.com/docs/home/home)                                   | US Securities          | 1998-now, 1s             | NA                    | OHLCV                 |
| [RiceQuant](https://www.ricequant.com/doc/rqdata/python/)                                     | CN Securities          | 2005-now, 1ms            | Account-specific      | OHLCV                 |
| [Tushare](https://tushare.pro/document/1?doc_id=131)                                          | CN Securities, A share | -now, 1 min              | Account-specific      | OHLCV                 |
| [WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/) | US Securities          | 2003-now, 1ms            | 5 requests each time  | Intraday Trades       |
| [YahooFinance](https://pypi.org/project/yfinance/)                                            | US Securities          | Frequency-specific, 1min | 2,000/hour            | OHLCV                 |

# Change Log
# New Contributors
# Frequently Asked Questions


