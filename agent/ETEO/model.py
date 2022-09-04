import torch
from torch import nn


# the state we have is squence of state, which I will see as the list whose components is the state,in that list, all the state is next to each other
# with respect to the time
class FCN_ETTO(torch.nn.Module):
    def __init__(self, length, input_dimension, nodes=128) -> None:
        # nodes is a list where the element reprensents the nodes on each layer
        super(FCN_ETTO).__init__()
        self.length = length
        self.nodes = nodes
        self.linear1 = nn.Linear(input_dimension, 128)
        self.linear2 = nn.Linear(128, 128)


class LSTM_ETEO(torch.nn.Module):
    def __init__(self, length, nodes, num_layer):
        super(LSTM_ETEO).__init__()
        self.length = length
        self.nodes = nodes
        self.num_layer = num_layer
