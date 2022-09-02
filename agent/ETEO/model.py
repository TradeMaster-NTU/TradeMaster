import torch
from torch import nn
# the state we have is squence of state, which I will see as the list whose components is the state,in that list, all the state is next to each other 
# with respect to the time 
class LSTM_ETEO(torch.nn.Module):
    def __init__(self,length,nodes,num_layer):
        super(LSTM_ETEO).__init__()
        self.length = length
        self.nodes=nodes
        self.num_layer=num_layer