# TradeMaster: An RL Platform for Trading
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-3713/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](Platform)
[![License](https://img.shields.io/github/license/ai-gamer/PRUDEX-Compass)](License)

<div align="center">
<img align="center" src=figure/Logo.png width="30%"/>
</div>

***
TradeMaster is a first-of-its kind, best-in-class open-source platform for quantitative trading (QT) empowered by reinforcement learning (RL).

It covers the full pipeline for the design, implementation, evaluation and deployment of RL-based trading methods. It contains: 1) a toolkit for efficient data collection, preprocessing and analysis; 2) high-fidelity data-driven market simulator for mainstream QT tasks (e.g., portfolio management and algorithmic trading); 3) standard implementation of over 10 novel FinRL methods; 4) a systematic evaluation benchmark called PRUDEX-Compass.

## Outline

- [TradeMaster: An RL Platform for Trading](#trademaster-an-rl-platform-for-trading)
  - [Outline](#outline)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Tutorial](#tutorial)
  - [Results and Visualization](#results-and-visualization)
  - [Model Zoo](#model-zoo)
  - [Supported Data Source](#supported-data-source)
  - [How to Use Your Own Data](#how-to-use-your-own-data)
  - [File Structure](#file-structure)
  - [Publications](#publications)
  - [Contact](#contact)

## Overview
<div align="center">
<img align="center" src=figure/Architecture.jpg width="70%"/>
</div>

TradeMaster could be beneficial to a wide range of communities including leading trading firms, startups, financial service providers and personal investors. We hope TradeMaster can make a change for the whole pipeline of FinRL to prevent untrustworthy results and lead successful industry deployment.

## Installation
To install the dependency of `TradeMaster`, run the command:
```
pip install -r requirements.txt
```

## Tutorial
We provide a tutorial for users to get start with.
|  Algorithm  | Dataset |                                                       Code link                                                       |                     Description                      |
| :---------: | :-----: | :-------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------: |
| Classic RL  |   FX    |   [tutorial](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/ClassicalRL_for_PM_on_FX.ipynb)    | Classic RL Algorithms for Portfolio Management on FX |
| DeepScalper | Bitcoin | [tutorial](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/DeepScalper_for_AT_on_Bitcoin.ipynb) |     DeepScalper for Algorithm Trading on Crypto      |
|    EIIE     |  DJ30   |      [tutorial](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/EIIE_for_PM_on_DJ30.ipynb)      |        EIIE for Portfolio Management on DJ30         |
|    IMIT     |  DJ30   |      [tutorial](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/IMIT_for_PM_on_DJ30.ipynb)      |  Investor Imitator for Portfolio Management on DJ30  |
|    SARL     |  DJ30   |      [tutorial](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/SARL_for_PM_on_DJ30.ipynb)      |        SARL for Portfolio Management on DJ30         |

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
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/master/tutorial/result/visualization/pp-1.png" width = 300 height = 170 />   </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/master/tutorial/visualization_data/USPP.PNG" width = 300 height = 170 /> </center></td>
        <td ><center><img src="https://github.com/TradeMaster-NTU/TradeMaster/blob/master/tutorial/visualization_data/FXPP.PNG" width = 300 height = 170 /> </center></td>
    </tr>
    <tr>
     <td align="center"><center>(a) All 4 datasets</center></td><td align="center"><center>(b) DJ30</center></td>      <td align="center"><center>(c) FX</center></td>                   
    </tr>
</table>


<!-- <div align="center">
  <img src="https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/result/visualization/pp-1.png" width = 300 height = 225 />
</div> -->

For more information of the usage of this part, please refer to this [tutorial](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/tutorial/Visualization.ipynb) and this [project](https://github.com/ai-gamer/PRUDEX-Compass)



## Model Zoo
[DeepScalper based on Pytorch (Shuo Sun et al, CIKM 22)](https://arxiv.org/abs/2201.09058)

[SARL based on Pytorch (Yunan Ye et al, AAAI 20)](https://arxiv.org/abs/2002.05780)

[Investor-Imitator based on Pytorch (Yi Ding et al, KDD 18)](https://www.kdd.org/kdd2018/accepted-papers/view/investor-imitator-a-framework-for-trading-knowledge-extraction)

[EIIE based on Pytorch (Jiang et al, 17)](https://arxiv.org/abs/1706.10059)

Classic RL based on Pytorch and Ray: 
[PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo) [A2C](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#a3c) [SAC](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac) [DDPG](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ddpg) [DQN](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn) [PG](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#pg) [TD3](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ddpg)

## Supported Data Source
| Dataset |                    Data Source                     |     Type      |     Range and Frequency     | Raw Data | Preprocessed Data |                                            Datasheet                                             |
| :-----: | :------------------------------------------------: | :-----------: | :-------------------------: | :------: | :---------------: | :----------------------------------------------------------------------------------------------: |
|  DJ30   | [YahooFinance](https://pypi.org/project/yfinance/) |   US Stock    | 2012/01/01-2021/12/31, 1day |  OHLCV   | Prices&Indicators |   [DJ30](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/data/data/dj30/DJ30.pdf)   |
|   FX    |    [Kaggle](https://pypi.org/project/yfinance/)    |      FX       | 2000/01/01-2019/12/31, 1day |  OHLCV   | Prices&Indicators |   [FX](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/data/data/exchange/FX.pdf)   |
| Crypto  |    [Kaggle](https://pypi.org/project/yfinance/)    |    Crypto     | 2013/04/29-2021/07/06, 1day |  OHLCV   | Prices&Indicators | [Crypto](https://github.com/qinmoelei/TradeMaster_reframe/blob/master/data/data/exchange/FX.pdf) |
|  SZ50   |      [JoinQuant](https://www.joinquant.com/)       | CN Securities | 2009/01/02-2021-01-01, 1day |  OHLCV   | Prices&Indicators |   [SZ50](https://github.com/qinmoelei/TradeMaster_reframe/tree/master/data/data/sz50/SZ50.pdf)   |

## How to Use Your Own Data
TradeMaster supports financial data with open, high, low, close, volume (OHLCV) raw informations as:

<div align="center">
<img align="center" src=figure/ohlcv.jpg width="70%"/>
</div>

we compute 10 technical indicators to describe the financial markets:

<div align="center">
<img align="center" src=figure/feature.jpg width="40%"/>
</div>

Users can adapt their data with prefered features by changing the data loading and feature calculation part with corresponding input and output size.
We plan to support limit order book (LOB) and altervative data such as text and graph in the future.

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
[Deep Reinforcement Learning for Quantitative Trading: Challenges and Opportunities](https://ieeexplore.ieee.org/abstract/document/9779600) *IEEE Intelligent System*

[DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities](https://arxiv.org/abs/2201.09058) *CIKM 22*

[Reinforcement Learning for Quantitative Trading (Survey)](https://arxiv.org/abs/2109.13851)

[PRUDEX-Compass: Towards Systematic Evaluation of Reinforcement Learning in Financial Markets](https://openreview.net/forum?id=Vhb-awTdHCh)

[Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio Management](https://ojs.aaai.org/index.php/AAAI/article/view/16142) *AAAI 21*
## Contact
