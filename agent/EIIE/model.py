import torch
from torch import nn
import numpy as np


class EIIE_con(torch.nn.Module):
    def __init__(self, in_channels, out_channels, length, kernel_size=3):
        super(EIIE_con, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.length = length
        self.act = torch.nn.ReLU(inplace=False)
        self.con1d = nn.Conv1d(self.in_channels,
                               self.out_channels,
                               kernel_size=3)
        self.con2d = nn.Conv1d(self.out_channels,
                               1,
                               kernel_size=self.length - self.kernel_size + 1)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.con1d(x)
        x = self.act(x)
        x = self.con2d(x)
        x = self.act(x)
        x = self.con3d(x)
        x = x.view(-1)

        # self.linear2 = nn.Linear(len(x), len(x) + 1)
        # x = self.linear2(x)
        x = torch.cat((x, self.para), dim=0)
        x = torch.softmax(x, dim=0)

        return x


class EIIE_lstm(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(EIIE_lstm, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, 1)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[:, -1, :]).view(-1, 1, 1)
        x = self.con3d(x)
        x = x.view(-1)
        x = torch.cat((x, self.para), dim=0)
        x = torch.softmax(x, dim=0)
        return x


class EIIE_rnn(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(EIIE_rnn, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.rnn = nn.RNN(input_size=n_features,
                          hidden_size=self.n_hidden,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.linear = nn.Linear(self.n_hidden, 1)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):
        lstm_out, _ = self.rnn(x)
        x = self.linear(lstm_out[:, -1, :]).view(-1, 1, 1)
        x = self.con3d(x)
        x = x.view(-1)
        x = torch.cat((x, self.para), dim=0)
        x = torch.softmax(x, dim=0)
        return x


class EIIE_critirc(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(EIIE_critirc, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, 1)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x, a):
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[:, -1, :]).view(-1, 1, 1)
        x = self.con3d(x)
        x = x.view(-1)
        x = torch.cat((x, self.para, a), dim=0)
        x = torch.nn.ReLU(inplace=False)(x)
        number_nodes = len(x)
        self.linear2 = nn.Linear(number_nodes, 1)
        x = self.linear2(x)
        return x


def normalization(x: torch.Tensor):
    x = x / torch.abs(x).sum()
    return x


if __name__ == "__main__":
    x = torch.randn(12)
    print(x)
    print(normalization(x))
