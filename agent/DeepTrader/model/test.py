from TCN import Chomp1d, TemporalBlock, TemporalConvNet
import sys
import torch

sys.path.append(".")
from env.portfolio_management.portfolio_for_deeptrader import *
from env.portfolio_management.portfolio_for_deeptrader import Tradingenv

a = Tradingenv(vars(args))
state = a.reset()
print(state.shape)
net = TemporalConvNet(16, [12, 12, 12, 12])
print(net(torch.from_numpy(state).to(torch.float32)).shape)
