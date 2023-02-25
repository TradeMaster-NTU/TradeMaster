# Publication
[Deep Reinforcement Learning for Quantitative Trading: Challenges and Opportunities](https://ieeexplore.ieee.org/abstract/document/9779600) *(IEEE Intelligent Systems 2022)*

[DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities](https://arxiv.org/abs/2201.09058) *(CIKM 2022)*

[Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio Management](https://ojs.aaai.org/index.php/AAAI/article/view/16142) *(AAAI 21)*

[Reinforcement Learning for Quantitative Trading (Survey)](https://arxiv.org/abs/2109.13851)

[PRUDEX-Compass: Towards Systematic Evaluation of Reinforcement Learning in Financial Markets](https://openreview.net/forum?id=Vhb-awTdHCh)

# File Structure
Here is the structure of the TradeMaster project. 

```
| TradeMaster
| ├── configs
| │   ├── base
| │   ├── algorithmic_trading
| │   ├── order_excution
| │   └── porfolio_management
| ├── data
| │   ├── algorithmic_trading          
| │   ├── order_excution          
| │   └──  porfolio_management
| ├── deploy
| │   ├── backend_client_test.py         
| │   ├── backend_client.py
| │   ├── backend_service_test.py  
| │   └── backend_service.py  
| ├── tools
| │   ├── algorithmic_trading          
| │   ├── MarketRegimeLabeling   
| │   ├── order_excution  
| │   ├── porfolio_management  
| │   ├── __init__.py 
| │   └── tmp.py      
| ├── tradmaster
| │   ├── __pycache         
| │   ├── agents   
| │   ├── datasets 
| │   ├── enviornments 
| │   ├── losses
| │   ├── nets
| │   ├── optimizers
| │   ├── pretrained
| │   ├── trainers
| │   ├── utils
| │   └── __init__.py     
| ├── unit_testing
| │   ├── test_agents\algorithmic_trading        
| │   ├── test_datasets
| │   ├── test_enviornments 
| │   ├── test_losses
| │   ├── test_nets
| │   ├── test_optimizers
| │   ├── test_trainers
| │   ├── __init__.py   
| │   └── test_score.py  
| ├── LICENSE
| ├── python3.9.yaml
| └── README.md
```

In the folder structure above:

- ``configs`` contains configuration files directory for agents trainng.
- ``data`` contains the data for training and testing.
- ``deploy`` contains the backend deploying scripts.
- ``tools`` contains training scripts.
- ``trademaster`` contains the files defining agents,enviornments, training losses, nets, optimizers and helper functions.
- ``unit_testing`` contains the components for testing.

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


