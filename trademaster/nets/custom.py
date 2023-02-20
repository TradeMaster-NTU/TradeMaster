from torch import nn
from .builder import NETS

@NETS.register_module()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()