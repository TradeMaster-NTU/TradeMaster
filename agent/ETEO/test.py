import numpy as np
import torch
from torch import nn
from random import sample

list_states = []
stacked_state = [np.random.rand(156, )] * 10
for state in stacked_state:
    state = torch.from_numpy(state).reshape(1, -1)
    list_states.append(state)
list_states = torch.cat(list_states, dim=0)
print(list_states.shape)
print(1 % 2)
print(sample(range(10), 10))
print(
    torch.normal(mean=torch.tensor([1, 50000]).float(),
                 std=torch.tensor([1, 1]).float()))
