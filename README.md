# TradeMaster: An RL Platform for Trading

<div align="center">
<img align="center" src=https://github.com/TradeMaster-NTU/TradeMaster/blob/main/figure/Logo.png width="25%"/> 

<div>&nbsp;</div>
  
[![Python 3.9](https://shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](Platform)
[![License](https://img.shields.io/github/license/TradeMaster-NTU/TradeMaster)](License)
[![Document](https://img.shields.io/badge/docs-latest-red)](https://trademaster.readthedocs.io/en/latest/)
[![GitHub stars](https://img.shields.io/github/stars/TradeMaster-NTU/TradeMaster?color=orange)](https://github.com/TradeMaster-NTU/TradeMaster/stargazers)

</div>

***
TradeMaster is a first-of-its kind, best-in-class __open-source platform__ for __quantitative trading (QT)__ empowered by __reinforcement learning (RL)__, which covers the __full pipeline__ for the design, implementation, evaluation and deployment of RL-based algorithms.

## :star: **What's NEW!**   :alarm_clock: 

| Update | Status |
| --                      | ------    |
| Release [Colab](https://colab.research.google.com/drive/10M3F6qF8qJ31eQkBR7B6OHhYCR1ZUlrp#scrollTo=4TKpEroeFdT4) Version | :speech_balloon: [Updated](https://github.com/TradeMaster-NTU/TradeMaster/pull/133) on 29 March 2023 |
| Incldue [HK Stock](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/data/portfolio_management/hstech30/HSTech30.pdf) and [Future](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/data/portfolio_management/Future/future.pdf) Datasets | :compass: Updated [#131](https://github.com/TradeMaster-NTU/TradeMaster/pull/131) [#132](https://github.com/TradeMaster-NTU/TradeMaster/pull/132) on 27 March 2023 |
| Support [Alpha158](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial8_Train_with_more_technical_indicators.ipynb) | :hammer: Updated [#123](https://github.com/TradeMaster-NTU/TradeMaster/pull/123) [#124](https://github.com/TradeMaster-NTU/TradeMaster/pull/124) on 20 March 2023 |
| Release TradeMaster 1.0.0 | :octocat: [Released v1.0.0](https://github.com/TradeMaster-NTU/TradeMaster/releases/tag/v1.0.0) on 5 March 2023 |

## Outline

- [TradeMaster: An RL Platform for Trading](#trademaster-an-rl-platform-for-trading)
  - [Outline](#outline)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Tutorial](#tutorial)
  - [Useful Script](#useful-script)
  - [Dataset](#dataset)
  - [Model Zoo](#model-zoo)
  - [Visualization Toolkit](#visualization-toolkit)
  - [File Structure](#file-structure)
  - [Publications](#publications)
  - [News](#news)
  - [Team](#team)
  - [Competition](#competition)
  - [Contact Us](#contact-us)

## Overview
<div align="center">
<img align="center" src=https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/architecture.jpg width="97%"/>
</div>
<br>

__TradeMaster__ is composed of 6 key modules: 1) multi-modality market data of different financial assets at multiple granularity; 2) whole data preprocessing pipeline; 3) a series of high-fidelity data-driven market simulators for mainstream QT tasks; 4) efficient implementations of over 13 novel RL-based trading algorithms; 5) systematic evaluation toolkits with 6 axes and 17 measures; 6) different interfaces for interdisciplinary users.


## Installation
Here are the installation tutorials for different operating systems and docker:
- [Installation on Linux/Windows/MacOS](https://github.com/TradeMaster-NTU/TradeMaster/tree/1.0.0/installation/requirements.md)
- [Installation with Docker](https://github.com/TradeMaster-NTU/TradeMaster/tree/1.0.0/installation/docker.md)

## Tutorial
We provide tutorials covering core features of TradeMaster for users to get start with.
|  Algorithm  | Dataset |   Market |                                                  Task                                                 |                     Code Link                      |
| :---------: | :-----: | :-----: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------: |
| EIIE | DJ30 | US Stock | Portfolio Management | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial1_EIIE.ipynb)|
| DeepScalper  |   BTC | Crypto | Intraday Trading | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial2_DeepScalper.ipynb) | 
| SARL | DJ30 | US Stock | Portfolio Management | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial3_SARL.ipynb)| 
| PPO  |  SSE 50  | China Stock | Portfolio Management | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial4_PPO.ipynb)|
| ETEO | Bitcoin | Crypto | Order Execution | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial5_ETEO.ipynb)|
| Double DQN | Bitcoin | Crypto | High Frequency Trading | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial6_DDQN.ipynb)|

We also provide a colab version of these tutorials that can be run directly. ([colab tutorial](https://colab.research.google.com/drive/10M3F6qF8qJ31eQkBR7B6OHhYCR1ZUlrp#scrollTo=4TKpEroeFdT4))


## Useful Script
- [Automatic hyperparameter tuning](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial7_auto_tuning.ipynb)
- [Market dynamic labeling](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/trademaster/evaluation/market_dynamics_labeling/example.ipynb)
- [Financial data imputation with diffusion models](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tools/missing_value_imputation/run.py)
- [Train RL agents with Alpha158 technical indicators](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/Tutorial8_Train_with_more_technical_indicators.ipynb)


## Dataset
| Dataset |                    Data Source                     |     Type      |           Range and Frequency            | Raw Data |                                                 Datasheet                                                 |
|:-------:| :------------------------------------------------: | :-----------: | :--------------------------------------: | :------: | :-------------------------------------------------------------------------------------------------------: |
| S&P500  | [Yahoo](https://pypi.org/project/yfinance/) |   US Stock    |       2000/01/01-2022/01/01, 1day        |  OHLCV   |         [SP500]()          |
|  DJ30   | [Yahoo](https://pypi.org/project/yfinance/) |   US Stock    |       2012/01/01-2021/12/31, 1day        |  OHLCV   |         [DJ30](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/dj30/DJ30.pdf)          |
|   BTC   |    [Kaggle](https://pypi.org/project/yfinance/)    |      Foreign Exchange       |     2000/01/01-2019/12/31, 1day        |  OHLCV   |         [FX](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/exchange/FX.pdf)          |
| Crypto  |    [Kaggle](https://pypi.org/project/yfinance/)    |    Crypto     |       2013/04/29-2021/07/06, 1day        |  OHLCV   |        [Crypto](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/BTC/Crypto.pdf)        |
|  SSE50  | [Yahoo](https://pypi.org/project/yfinance/) | China Stock |       2009/01/02-2021/01/01, 1day        |  OHLCV   |         [SSE50](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/sz50/SZ50.pdf)          |
| Bitcoin |                     [Binance](https://www.binance.com/)                     |    Crypto     | 2021/04/07-2021/04/19, 1min |   LOB    | [Binance](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/OE_BTC/limit_order_book.pdf) |
| Future |                     [AKshare](https://akshare.xyz/)                     |    Future     | 2023/03/07-2023/03/28, 5min |   OHLCV    | [Future](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/data/portfolio_management/Future/future.pdf) |
| HS30 |                     [AKShare](https://akshare.xyz/)                     |    HK Stock     | 1988/12/30-2023/03/27, 1day |   OHLCV    | [HS30](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/data/portfolio_management/hstech30/HSTech30.pdf) |


Dates are in YY/MM/DD format.


OHLCV: open, high, low, and close prices; volume: corresponding trading volume; LOB: Limit order book.

Users can download data of the above datasets from [Google Drive](https://drive.google.com/drive/folders/19Tk5ifPz1y8i_pJVwZFxaSueTLjz6qo3?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1njghvez53hD5v3WpLgCg0w) (extraction code:x24b)
## Model Zoo
TradeMaster provides efficient implementations of the following algorithms:

[DeepScalper based on Pytorch (Shuo Sun et al, CIKM 22)](https://arxiv.org/abs/2201.09058)

[OPD based on Pytorch (Fang et al, AAAI 21)](https://ojs.aaai.org/index.php/AAAI/article/view/16083)

[DeepTrader based on Pytorch (Wang et al, AAAI 21)](https://ojs.aaai.org/index.php/AAAI/article/view/16144) 

[SARL based on Pytorch (Yunan Ye et al, AAAI 20)](https://arxiv.org/abs/2002.05780)

[ETEO based on Pytorch (Lin et al, 20)](https://www.ijcai.org/Proceedings/2020/627?msclkid=a2b6ad5db7ca11ecb537627a9ca1d4f6)

[Investor-Imitator based on Pytorch (Yi Ding et al, KDD 18)](https://www.kdd.org/kdd2018/accepted-papers/view/investor-imitator-a-framework-for-trading-knowledge-extraction)

[EIIE based on Pytorch (Jiang et al, 17)](https://arxiv.org/abs/1706.10059)


Classic RL based on Pytorch and Ray: 
[PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo) [A2C](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#a3c) [Rainbow](https://docs.ray.io/en/releases-1.13.0/rllib/rllib-algorithms.html#dqn) [SAC](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac) [DDPG](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ddpg) [DQN](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn) [PG](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#pg) [TD3](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ddpg)

## Visualization Toolkit
TradeMaster provides many visualization toolkits for a systematic evaluation of RL-based quantitative trading methods. Please check this [paper](https://openreview.net/forum?id=JjbsIYOuNi) and [repository](https://github.com/TradeMaster-NTU/PRUDEX-Compass) for details. Some examples are as follows:

**PRIDE-Star** is a star plot containing normalized score of 8 key financial measures such total return (TR) and Sharpe ratio (SR) to evaluate profitability,risk-control and diversity:
<table align="center">
    <tr>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/visualization/A2C.jpg" width="95%" />   </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/visualization/DeepTrader.jpg" width="95%" /> </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/visualization/PPO.jpg" width="95%" /> </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/visualization/EIIE.jpg" width="95%" /> </center></td>
    </tr>
</table>

<div align="left">
<img align="center" src=https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/visualization/plot1.jpg width="100%"/>
</div>
<br>

<div align="left">
<img align="center" src=https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/visualization/plot2.jpg width="100%"/>
</div>
<br>


## File Structure

```
| TradeMaster
| ├── configs
| ├── data
| │   ├── algorithmic_trading 
| │   ├── high_frequency_trading  
| │   ├── order_excution          
| │   └── porfolio_management
| ├── deploy
| │   ├── backend_client.py
| │   ├── backend_client_test.py 
| │   └── backend_service.py        
| │   ├── backend_service_test.py  
| ├── docs
| ├── figure
| ├── installation
| │   ├── docker.md
| │   ├── requirements.md
| ├── tools
| │   ├── algorithmic_trading          
| │   ├── data_preprocessor
| │   ├── high_frequency_trading
| │   ├── market_dynamics_labeling
| │   ├── missing_value_imputation  
| │   ├── order_excution  
| │   ├── porfolio_management  
| │   ├── __init__.py      
| ├── tradmaster       
| │   ├── agents   
| │   ├── datasets 
| │   ├── enviornments 
| │   ├── evaluation 
| │   ├── imputation 
| │   ├── losses
| │   ├── nets
| │   ├── preprocessor
| │   ├── optimizers
| │   ├── pretrained
| │   ├── trainers
| │   ├── transition
| │   ├── utils
| │   └── __init__.py     
| ├── unit_testing
| ├── Dockerfile
| ├── LICENSE
| ├── README.md
| ├── pyproject.toml
| └── requirements.txt
```

## Publications
[PRUDEX-Compass: Towards Systematic Evaluation of Reinforcement Learning in Financial Markets](https://openreview.net/forum?id=JjbsIYOuNi) *(Transactions on Machine learning Research 2023)*

[Reinforcement Learning for Quantitative Trading (Survey)](https://dl.acm.org/doi/10.1145/3582560) *(ACM Transactions on Intelligent Systems and Technology 2023)*

[Deep Reinforcement Learning for Quantitative Trading: Challenges and Opportunities](https://ieeexplore.ieee.org/abstract/document/9779600) *(IEEE Intelligent Systems 2022)*

[DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities](https://arxiv.org/abs/2201.09058) *(CIKM 2022)*

[Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio Management](https://ojs.aaai.org/index.php/AAAI/article/view/16142) *(AAAI 21)*

## News
- [知乎][用强化学习在金融市场上赚钱？南洋理工发布全新基于强化学习的量化交易平台TradeMaster](https://zhuanlan.zhihu.com/p/614855780?utm_medium=social&utm_oi=798261254362927104&utm_psn=1620381293060993024&utm_source=wechat_session&wechatShare=1&s_r=0)
- [机器之心][南洋理工发布量化交易大师TradeMaster，涵盖15种强化学习算法](https://mp.weixin.qq.com/s/MTUOksGGgaWX6GkXZT6wwA)
- [运筹OR帷幄][量化交易大师TradeMaster](https://mp.weixin.qq.com/s/bAuvKD5QD3Lz8ZC60WWJLQ)
- [Medium][Introduction on TradeMaster](https://medium.com/@trademaster.ntu/introduction-to-trademaster-a-new-standard-of-reinforcement-learning-framework-for-quantitative-67f7133485e2)


## Team
- This repository is developed and maintained by [AMI](https://personal.ntu.edu.sg/boan/) group lead by [Prof Bo An](boan@ntu.edu.sg) at [Nanyang Technological University](https://www.ntu.edu.sg/).
- We have positions for software engineer, research associate and postdoc. If you are interested in working at the intersection of RL and quantitative trading, feel free to send us an email with your CV.

## Competition
[TradeMaster Cup 2022](https://codalab.lisn.upsaclay.fr/competitions/8440?secret_key=51d5952f-d68d-47d9-baef-6032445dea01)

## Contact Us
If you have any further questions of this project, please contact [TradeMaster.NTU@gmail.com](TradeMaster.NTU@gmail.com)
