# Tutorial 6: High Frequency Trading with Double DQN

High Frequency Trading is a fundamental quantitative trading task, where traders actively buy/sell one pre-selected financial periodically in seconds with the consideration of order execution.

HFT_DDQN use a decayed supervised regulator genereated from the real q table based on the future price information and a double q network to optimizer the portfit margine.


In this notebook, we implement the training and testing process of HFTDDQN based on the TradeMaster framework.

[Tutorial6_HFT](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/tutorial/Tutorial6_DDQN.ipynb)

And this is the script for training and testing.

[train.py](https://github.com/TradeMaster-NTU/TradeMaster/blob/1.0.0/tools/high_frequency_trading/train.py)