import torch
import torch.nn as nn

from .builder import NETS
from .custom import Net


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


@NETS.register_module()
class ETEOStacked(Net):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, time_steps=10, explore_rate=0.25):
        # nodes is a list where the element reprensents the nodes on each layer
        super(ETEOStacked, self).__init__()

        self.net = build_mlp(dims=[state_dim * time_steps, *dims])

        self.act_linear_volume = nn.Linear(dims[-1], 2)
        self.act_linear_price = nn.Linear(dims[-1], 2)
        self.v_linear = nn.Linear(dims[-1], 1)

        # init weights
        self.net.apply(self.init_weights)
        self.act_linear_volume.apply(self.init_weights)
        self.act_linear_price.apply(self.init_weights)
        self.v_linear.apply(self.init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)

        x = self.net(x)

        action_volume = self.act_linear_volume(x)
        action_price = self.act_linear_price(x)
        v = self.v_linear(x)

        return action_volume, action_price, v

    def init_weights(self, m):
        # init linear
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.zero_()
