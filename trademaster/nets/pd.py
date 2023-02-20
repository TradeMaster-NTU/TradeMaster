import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import NETS
from .custom import Net


@NETS.register_module()
class PDNet(Net):
    def __init__(self, input_feature, hidden_size, private_feature):
        super(PDNet, self).__init__()
        self.rnn_public = nn.RNN(input_feature,
                                 hidden_size,
                                 num_layers=1,
                                 batch_first=True)
        self.rnn_private = nn.RNN(private_feature,
                                  hidden_size,
                                  num_layers=1,
                                  batch_first=True)
        self.L = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.mu = nn.Linear(hidden_size * 2, 1)
        self.sigma = nn.Linear(hidden_size * 2, 1)
        self.V = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, private_state):
        _, x = self.rnn_public(x)
        _, p = self.rnn_private(private_state)
        x = x.squeeze(0)
        p = p.squeeze(0)
        x = torch.concat([x, p], dim=1)
        x = self.L(x)

        mu = torch.sigmoid(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 0.001
        V = self.V(x)
        return mu, sigma, V

    def get_V(self, x, private_state):
        mu, sigma, V = self.forward(x, private_state)
        return V
