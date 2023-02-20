import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net
from trademaster.utils import build_conv2d
from torch import Tensor

@NETS.register_module()
class EIIEConv(Net):
    def __init__(self,
                 input_dim,
                 output_dim = 1,
                 time_steps = 10,
                 kernel_size = 3,
                 dims = (32, )):
        super(EIIEConv, self).__init__()

        self.kernel_size = kernel_size
        self.time_steps = time_steps

        self.net = build_conv2d(
            dims=[input_dim, *dims, output_dim],
            kernel_size=[(1, self.kernel_size), (1, self.time_steps - self.kernel_size + 1)]
        )
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x): # (batch_size, num_seqs, action_dim, time_steps, state_dim)
        if len(x.shape) > 4:
            x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        x = x.view(x.shape[0], -1)

        para = self.para.repeat(x.shape[0], 1)
        x = torch.cat((x, para), dim=1)
        x = torch.softmax(x, dim=1)
        return x

@NETS.register_module()
class EIIECritic(Net):
    def __init__(self,
                 input_dim,
                 action_dim,
                 output_dim=1,
                 time_steps=10,
                 num_layers= 1,
                 hidden_size = 32,
                 ):
        super(EIIECritic, self).__init__()

        self.time_steps = time_steps

        self.lstm = nn.LSTM(input_size=input_dim * time_steps,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear1 = nn.Linear(hidden_size, output_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(2 * (action_dim + 1), 1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x, a):
        if len(x.shape) >= 4:
            x = x.view(x.shape[0], x.shape[1], -1)
        lstm_out, _ = self.lstm(x)
        x = self.linear1(lstm_out)

        x = self.act(x)

        x = x.view(x.shape[0], -1)
        para = self.para.repeat(x.shape[0], 1)

        x = torch.cat((x, para, a), dim=1)
        # x = self.linear2(x)
        x = x.mean(dim = 1, keepdim=True)
        return x