import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_reg(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP_reg, self).__init__()
        self.input_size = input_size
        self.n_hidden = hidden_size
        self.act = torch.nn.LeakyReLU()
        self.linear1 = nn.Linear(self.input_size, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.linear3 = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x.squeeze()


class MLP_cls(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_cls, self).__init__()
        self.affline1 = nn.Linear(input_size, hidden_size)
        self.affline2 = nn.Linear(hidden_size, output_size)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # x = torch.nn.Sigmoid()(x)

        x = self.affline1(x)
        x = torch.nn.Sigmoid()(x)
        action_scores = self.affline2(x).unsqueeze(0)

        return F.softmax(action_scores, dim=1)