import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import NETS
from .custom import Net


@NETS.register_module()
class LSTMClf(Net):
    def __init__(self, n_features, layer_num, n_hidden):
        super(LSTMClf, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return F.softmax(self.linear(lstm_out[:, -1, :]))


@NETS.register_module()
class mLSTMClf(Net):
    def __init__(self, n_features, layer_num, n_hidden, tic_number):
        super(mLSTMClf, self).__init__()
        self.tic_number = tic_number
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.lstm_list = nn.ModuleList([
                       nn.LSTM(input_size=n_features,
                               hidden_size=self.n_hidden,
                               num_layers=self.n_layers,
                               batch_first=True).cuda()
                                       ] * tic_number)
        self.linear = nn.Linear(self.n_hidden * tic_number * 2, tic_number)

    def forward(self, x):
        # print(x.shape)
        ch_out = []
        for i in range(self.tic_number):
            tic_in = x[:, i, :, :]
            out, (h, c) = self.lstm_list[i](tic_in)
            ch_out.append(h.squeeze(0))
            ch_out.append(c.squeeze(0))

        ch_out = torch.cat(ch_out, dim=1).cuda()
        # print(ch_out.shape)
        y = self.linear(ch_out)
        y = torch.sigmoid(y)
        return y
