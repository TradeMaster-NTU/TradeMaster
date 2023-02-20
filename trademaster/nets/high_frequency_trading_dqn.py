import torch
import torch.nn as nn

from .builder import NETS
from .custom import Net
from torch import Tensor
import torch.nn.functional as F

@NETS.register_module()
class HFTQNet(Net):
    def __init__(self, dims: int, state_dim: int, action_dim: int, explore_rate = 0.25,max_punish=0):
        super(HFTQNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, dims)
        self.fc2 = nn.Linear(2 * dims, dims)
        self.out = nn.Linear(dims, action_dim)

        self.embedding = nn.Embedding(action_dim, dims)

        self.register_buffer('max_punish', torch.tensor(max_punish))
        self.explore_rate = explore_rate
        self.action_dim = action_dim
    def forward(self, state: torch.tensor, previous_action: torch.tensor,
                avaliable_action: torch.tensor):
        state_hidden = F.relu(self.fc1(state))
        previous_action_hidden = self.embedding(previous_action)
        information_hidden = torch.cat([state_hidden, previous_action_hidden],
                                       dim=1)
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action + (avaliable_action - 1) *self.max_punish
        return masked_action
    def get_action(self, state: Tensor,previous_action: Tensor,avaliable_action: Tensor) -> Tensor:  # return the index [int] of discrete action for exploration
        if self.explore_rate < torch.rand(1):
            action = self.forward(state,previous_action,avaliable_action).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action
    def get_action_test(self, state: Tensor,previous_action: Tensor,avaliable_action: Tensor) -> Tensor:
        action = self.forward(state,previous_action,avaliable_action).argmax(dim=1, keepdim=True)
        return action
        
    
    def init_weights(self, m):
        # init linear
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.zero_()