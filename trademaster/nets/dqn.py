import torch
import torch.nn as nn

from .builder import NETS
from .custom import Net
from torch import Tensor
from trademaster.utils import build_mlp

@NETS.register_module()
class QNet(Net):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, explore_rate = 0.25):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_rate = explore_rate
        self.action_dim = action_dim

        # init weights
        self.net.apply(self.init_weights)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # Q values for multiple actions

    def get_action(self, state: Tensor) -> Tensor:  # return the index [int] of discrete action for exploration
        if self.explore_rate < torch.rand(1):
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action

    def init_weights(self, m):
        # init linear
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.zero_()