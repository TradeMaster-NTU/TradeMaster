# TradeMaster: An RL Platform for Trading

<div align="center">
<img align="center" src=https://github.com/TradeMaster-NTU/TradeMaster/blob/main/figure/Logo.png width="25%"/> 

<div>&nbsp;</div>
  
[![Python 3.9](https://shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](Platform)
[![License](https://img.shields.io/github/license/TradeMaster-NTU/TradeMaster)](License)
[![Document](https://img.shields.io/badge/docs-latest-red)](https://trademaster.readthedocs.io/en/latest/)

</div>

***
TradeMaster is a first-of-its kind, best-in-class __open-source platform__ for __quantitative trading (QT)__ empowered by __reinforcement learning (RL)__, which covers the __full pipeline__ for the design, implementation, evaluation and deployment of RL-based algorithms.

## :star: **What's NEW!**   :alarm_clock: 

| Update | Status |
| --                      | ------    |
| Release TradeMaster 1.0.0 | :octocat: [Released] on Dec 9, 2022 |

## Outline

- [TradeMaster: An RL Platform for Trading](#trademaster-an-rl-platform-for-trading)
  - [Outline](#outline)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Tutorial](#tutorial)
  - [Useful Script](#useful-script)
  - [Dataset](#dataset)
  - [Model Zoo](#model-zoo)
  - [Results and Visualization](#results-and-visualization)
  - [File Structure](#file-structure)
  - [Publications](#publications)
  - [Contact](#contact)
  - [Join Us](#join-us)
  - [Competition](#competition)

## Overview
<div align="center">
<img align="center" src=https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/figure/architecture.jpg width="85%"/>
</div>
<br>

__TradeMaster__ is composed of 6 key modules: 1) multi-modality market data of different financial assets at multiple granularity; 2) whole data preprocessing pipeline; 3) a series of high-fidelity data-driven market simulators for mainstream QT tasks; 4) efficient implementations of over 13 novel RL-based trading algorithms; 5) systematic evaluation toolkits with 6 axes and 17 measures; 6) different interfaces for interdisciplinary users.


## Installation
We provide a video tutorial of using docker to build a proper environment of running this project.

[![Video Tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/installation/cover(1).png)](https://www.youtube.com/watch?v=uo80u1byGRc)

To help you better understand the step discribed in the video, Here are the installation tutorials for different operating systems:
- [Prerequisite](https://github.com/TradeMaster-NTU/TradeMaster/tree/1.0.0/installation/prerequisite.md)
- [Installation using Requirements](https://github.com/TradeMaster-NTU/TradeMaster/tree/1.0.0/installation/requirements.md)
- [Installation using Docker](https://github.com/TradeMaster-NTU/TradeMaster/tree/1.0.0/installation/docker.md)

## Tutorial
We provide tutorials for users to get start with.
|  Algorithm  | Dataset |                                                     Task                                                 |                     Code Link                      |
| :---------: | :-----: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------: |
| DeepScalper  |   Bitcoin |  Intraday Trading | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/DeepScalper_IT_Bitcoin.ipynb) | 
| EIIE | Dow Jones 30 | Portfolio Management | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/EIIE_PM_DJ30.ipynb)|
| SARL |  S&P 500 | Portfolio Management | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/EIIE_for_PM_on_SP500.ipynb)| 
| PPO  |  SSE 50  | Portfolio Management | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/PPO_PM_SSE50.ipynb)|
| ETTO |  | Order Execution | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/ETTO_OE_Bitcoin.ipynb)|
| Double DQN | Bitcoin | High Frequency Trading | [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tutorial/DDQN_HFT_Bitcoin.ipynb)|


## Useful Script
- Auto RL
- Automatic market style recognition [(link)](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/MarketRegimeLabeling/README.md)
- [CSDI](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html) for financial data imputation [(link)](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/CSDI/README.md)
- [Colab Version](https://colab.research.google.com/drive/10M3F6qF8qJ31eQkBR7B6OHhYCR1ZUlrp#scrollTo=4TKpEroeFdT4): Use Google Colab resource to run TradeMaster on Cloud  


## Dataset
| Dataset |                    Data Source                     |     Type      |           Range and Frequency            | Raw Data |                                                 Datasheet                                                 |
| :-----: | :------------------------------------------------: | :-----------: | :--------------------------------------: | :------: | :-------------------------------------------------------------------------------------------------------: |
|  SP500   | [YahooFinance](https://pypi.org/project/yfinance/) |   US Stock    |       2000/01/01-2022/01/01, 1day        |  OHLCV   |         [SP500]()          |
|  DJ30   | [YahooFinance](https://pypi.org/project/yfinance/) |   US Stock    |       2012/01/01-2021/12/31, 1day        |  OHLCV   |         [DJ30](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/dj30/DJ30.pdf)          |
|   FX    |    [Kaggle](https://pypi.org/project/yfinance/)    |      FX       |       2000/01/01-2019/12/31, 1day        |  OHLCV   |         [FX](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/exchange/FX.pdf)          |
| Crypto  |    [Kaggle](https://pypi.org/project/yfinance/)    |    Crypto     |       2013/04/29-2021/07/06, 1day        |  OHLCV   |        [Crypto](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/BTC/Crypto.pdf)        |
|  SZ50   | [YahooFinance](https://pypi.org/project/yfinance/) | CN Securities |       2009/01/02-2021-01-01, 1day        |  OHLCV   |         [SZ50](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/sz50/SZ50.pdf)          |
| Bitcoin |                     [Kaggle]()                     |    Crypto     | 2021-04-07 11:33-2021-04-19 09:54 , 1min |   LOB    | [Bitcoin](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/data/OE_BTC/limit_order_book.pdf) |

OHLCV: open, high, low, and close prices; volume: corresponding trading volume

## Model Zoo
[DeepScalper based on Pytorch (Shuo Sun et al, CIKM 22)](https://arxiv.org/abs/2201.09058)

[OPD based on Pytorch (Fang et al, AAAI 21)](https://ojs.aaai.org/index.php/AAAI/article/view/16083)

[DeepTrader based on Pytorch (Wang et al, AAAI 21)](https://ojs.aaai.org/index.php/AAAI/article/view/16144) 

[SARL based on Pytorch (Yunan Ye et al, AAAI 20)](https://arxiv.org/abs/2002.05780)

[ETTO based on Pytorch (Lin et al, 20)](https://www.ijcai.org/Proceedings/2020/627?msclkid=a2b6ad5db7ca11ecb537627a9ca1d4f6)

[Investor-Imitator based on Pytorch (Yi Ding et al, KDD 18)](https://www.kdd.org/kdd2018/accepted-papers/view/investor-imitator-a-framework-for-trading-knowledge-extraction)

[EIIE based on Pytorch (Jiang et al, 17)](https://arxiv.org/abs/1706.10059)


Classic RL based on Pytorch and Ray: 
[PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo) [A2C](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#a3c) [SAC](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac) [DDPG](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ddpg) [DQN](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn) [PG](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#pg) [TD3](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ddpg)

## Results and Visualization
The evaluation module of TradeMaster is mainly based on [PRUDEX-Compass](https://github.com/ai-gamer/PRUDEX-Compass), a systematic evaluation toolkit of FinRL methods with 6 axes and 17 measures. We show some results here:

**PRUDEX-Compass** provides an intuitive visual means to give readers a sense of comparability and positioning of FinRL methods. The inner level maps out the relative strength of FinRL methods in terms of each axis, whereas the outer level provides a compact way to visually assess which set-up and evaluation measures are practically reported to point out how comprehensive the evaluation are for FinRL algorithms.

<div align="center">
  <img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/PRUDEX.jpg" width = 500 height = 400 />
</div>

**PRIDE-Star** is a star plot to evaluate profitability,risk-control and diversity. It contains the normalized score of 8 measures.

<table align="center">
    <tr>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/A2C.PNG" width = 220 height = 200 />   </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/PPO.PNG" width = 220 height = 200 /> </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/SAC.PNG" width = 220 height = 200 /> </center></td>
    </tr>
    <tr>
     <td align="center"><center>(a) A2C</center></td><td align="center"><center>(b) PPO</center></td>      <td align="center"><center>(c) SAC</center></td>                   
    </tr>
</table>




**Rank distribution**
plot is a bar plot, where the i-th column in the rank distribution shows the probability that a given method is assigned rank i in the corresponding metrics.

<table align="center">
    <tr>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/result/visualization/rank-1.png" width = 300 height = 170 />   </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/USrank.PNG" width = 300 height = 170 /> </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/FXrank.PNG" width = 300 height = 170 /> </center></td>
    </tr>
    <tr>
     <td align="center"><center>(a) All 4 datasets</center></td><td align="center"><center>(b) DJ30</center></td>      <td align="center"><center>(c) FX</center></td>                   
    </tr>
</table>

<!-- <div align="center">
  <img src="https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/result/visualization/rank-1.png" width = 300 height = 225 />
</div> -->

**Performance profile** reports FinRL methods' score distribution of all runs across the different financial markets that are statistically unbiased and more robust to outliers.


<table align="center">
    <tr>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/result/visualization/pp-1.png" width = 300 height = 170 />   </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/USPP.PNG" width = 300 height = 170 /> </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/visualization_data/FXPP.PNG" width = 300 height = 170 /> </center></td>
    </tr>
    <tr>
     <td align="center"><center>(a) All 4 datasets</center></td><td align="center"><center>(b) DJ30</center></td>      <td align="center"><center>(c) FX</center></td>                   
    </tr>
</table>


<!-- <div align="center">
  <img src="https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/result/visualization/pp-1.png" width = 300 height = 225 />
</div> -->

For more information of the usage of this part, please refer to this [tutorial](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/Visualization.ipynb) and this [project](https://github.com/ai-gamer/PRUDEX-Compass)

## File Structure
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

## Publications
[PRUDEX-Compass: Towards Systematic Evaluation of Reinforcement Learning in Financial Markets](https://openreview.net/forum?id=JjbsIYOuNi) *(2023)*

[Reinforcement Learning for Quantitative Trading (Survey)](https://dl.acm.org/doi/10.1145/3582560) *(ACM Transactions on Intelligent Systems and Technology 2023)*

[Deep Reinforcement Learning for Quantitative Trading: Challenges and Opportunities](https://ieeexplore.ieee.org/abstract/document/9779600) *(IEEE Intelligent Systems 2022)*

[DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities](https://arxiv.org/abs/2201.09058) *(CIKM 2022)*

[Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio Management](https://ojs.aaai.org/index.php/AAAI/article/view/16142) *(AAAI 21)*


## Team
- This repository is developed and maintained by [AMI](https://personal.ntu.edu.sg/boan/) group at [Nanyang Technological University](https://www.ntu.edu.sg/).
- We have positions for software engineer, research associate and postdoc. If you are interested in working at the intersection of RL and quantitative trading, feel free to send an email to [Prfo An](boan@ntu.edu.sg) with your CV.


## Competition
[TradeMaster Cup 2022](https://codalab.lisn.upsaclay.fr/competitions/8440?secret_key=51d5952f-d68d-47d9-baef-6032445dea01)
