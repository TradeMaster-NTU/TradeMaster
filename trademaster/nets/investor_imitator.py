import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import NETS
from .custom import Net


@NETS.register_module()
class MLPReg(Net):
    def __init__(self, input_dim, dims, output_dim=1):
        super(MLPReg, self).__init__()
        self.input_dim = input_dim
        self.n_hidden = dims[0]
        self.output_dim = output_dim
        self.act = torch.nn.LeakyReLU()
        self.linear1 = nn.Linear(self.input_dim, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.linear3 = nn.Linear(self.n_hidden, output_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x.squeeze()

@NETS.register_module()
class MLPCls(Net):
    def __init__(self, input_dim, dims, output_dim=1):
        super(MLPCls, self).__init__()
        self.n_hidden = dims[0]
        self.affline1 = nn.Linear(input_dim, self.n_hidden)
        self.affline2 = nn.Linear(self.n_hidden, output_dim)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # x = torch.nn.Sigmoid()(x)
        x = self.affline1(x)
        x = torch.nn.Sigmoid()(x)
        action_scores = self.affline2(x).unsqueeze(0)

        return F.softmax(action_scores, dim=1)
