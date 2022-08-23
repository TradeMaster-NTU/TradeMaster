import torch  # 导入torch
import torch.nn as nn  # 导入torch.nn
import torch.nn.functional as F  # 导入torch.nn.functional


class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        self.out_a = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        actions_value = self.out(x)
        a_task = self.out_a(x)
        return actions_value, a_task
