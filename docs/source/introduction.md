# Introduction

## Architecture

TradeMaster could be beneficial to a wide range of communities including leading trading firms, startups, financial service providers and personal investors. We hope TradeMaster can make a change for the whole pipeline of FinRL to prevent untrustworthy results and lead successful industry deployment. \
Architecture of Trademaster framework could be visualizaed by the figure below.
- Level 1 Simulation

  Market Data and User Preference are fed into the network for data processing. In this step, data preprocessing, mining, augmentation, feature selection and behaviour cloning are performed to provide evident simulation of noisy real-world financial market.

- Level 2 Algorithm 
  
  A collection of various reinforcement learning algotithms are developed to provide feasible solution to different financial tasks like Algorithm Trading, Order Excuction and Porforlio Management. The RL algorithms will generate strategy to maximize user profit.
  
- Level 3 Evaluation

  TradeMaster is evaluated in multiple dimenstions. Financial metrics like profit and risk metrics are applied. Additionally, decision tree and shapley value are used to evaluate the explainability of the model. Variability and Alpha decay are used for reliability evaluation.

<div align="center">
<img align="center" src=../../figure/Architecture.jpg width="70%"/>
</div>

## Supported Trading Scenario
## Model Zoo

