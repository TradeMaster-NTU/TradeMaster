import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class RNN(nn.Module):
    def __init__(self, input_size, seq_len, num_rnn_layer, hidden_rnn,
                 rnn_mode):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_layer = num_rnn_layer
        self.hidden_rnn = hidden_rnn

        self.rnn_mode = rnn_mode
        if self.rnn_mode == 'lstm':
            self.rnn = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_rnn,
                               num_layers=self.num_layer,
                               batch_first=True,
                               bidirectional=False,
                               dropout=0.3)
        elif self.rnn_mode == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size,
                              hidden_size=self.hidden_rnn,
                              num_layers=self.num_layer,
                              batch_first=True,
                              bidirectional=False,
                              dropout=0.3)

        self.cx = torch.zeros(self.num_layer, 1,
                              self.hidden_rnn).float().cuda()
        self.hx = torch.zeros(self.num_layer, 1,
                              self.hidden_rnn).float().cuda()

    def reset_hidden_state(self, done=True):
        if done == True:
            ### hx/cx：[num_layer, batch, hidden_len] ###
            self.cx = torch.zeros(self.num_layer, 1,
                                  self.hidden_rnn).float().cuda()
            self.hx = torch.zeros(self.num_layer, 1,
                                  self.hidden_rnn).float().cuda()
        else:
            self.cx = self.cx.data.float().cuda()
            self.hx = self.hx.data.float().cuda()

    def forward(self, x, hidden_states=None):
        if self.rnn_mode == 'lstm':
            if hidden_states == None:  #agent與env互動select.action時走此流程，並且hidden_state會傳入下一step。
                out, (hx, cx) = self.rnn(x, (self.hx, self.cx))
                self.hx = hx
                self.cx = cx
            else:  # 在update policy時走此流程，並且每step都讓hidden_states歸0 (詳見rdpg.py的update_policy)
                out, (hx, cx) = self.rnn(x, hidden_states)

            xh = hx[self.num_layer - 1, :, :]  #注意是用hidden_state來接fc
            return xh, (hx, cx)

        elif self.rnn_mode == 'gru':
            if hidden_states == None:  #agent與env互動select.action時走此流程，並且hidden_state會傳入下一step。
                out, hx = self.rnn(x, self.hx)
                self.hx = hx
            else:  # 在update policy時走此流程，並且每step都讓hidden_states歸0 (詳見rdpg.py的update_policy)
                out, hx = self.rnn(x, hidden_states)

            xh = hx[self.num_layer - 1, :, :]  #注意是用hidden_state來接fc
            return xh, hx


class Actor(nn.Module):
    def __init__(self,
                 init_w,
                 hidden_rnn,
                 hidden_fc1,
                 hidden_fc2,
                 hidden_fc3,
                 nb_actions=2):
        super(Actor, self).__init__()
        nb_actions = 2
        init_w = init_w
        self.hidden_rnn = hidden_rnn
        self.hidden_fc1 = hidden_fc1
        self.hidden_fc2 = hidden_fc2
        self.hidden_fc3 = hidden_fc3

        self.fc1 = nn.Linear(self.hidden_rnn, self.hidden_fc1)
        self.fc2 = nn.Linear(self.hidden_fc1, self.hidden_fc2)
        self.fc3 = nn.Linear(self.hidden_fc2, self.hidden_fc3)
        self.fc4 = nn.Linear(self.hidden_fc3, nb_actions)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        a = self.soft(x)
        return a


class Critic(nn.Module):
    def __init__(
        self,
        init_w,
        input_size,
        seq_len,
        num_rnn_layer,
        hidden_rnn,
        hidden_fc1,
        hidden_fc2,
        hidden_fc3,
        Reward_max_clip,
        discount,
    ):
        super(Critic, self).__init__()
        nb_actions = 2
        init_w = init_w
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_layer = num_rnn_layer
        self.hidden_rnn = hidden_rnn
        self.hidden_fc1 = hidden_fc1
        self.hidden_fc2 = hidden_fc2
        self.hidden_fc3 = hidden_fc3
        self.R_max = Reward_max_clip
        self.gamma = discount

        self.fc1h = nn.Linear(self.hidden_rnn, self.hidden_fc1)
        self.fc1a = nn.Linear(nb_actions, self.hidden_fc1)
        self.fc2 = nn.Linear(self.hidden_fc1, self.hidden_fc2)
        self.fc3 = nn.Linear(self.hidden_fc2, self.hidden_fc3)
        self.fc4 = nn.Linear(self.hidden_fc3, 1)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1h.weight.data = fanin_init(self.fc1h.weight.data.size())
        self.fc1a.weight.data = fanin_init(self.fc1a.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        xh, a = xs
        xh = self.fc1h(xh)

        xa = self.fc1a(a)
        x = self.relu(xh + xa)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        c = self.fc4(x)
        # c = self.tanh(c) * (self.R_max / (1-self.gamma)) *0.2
        return c
