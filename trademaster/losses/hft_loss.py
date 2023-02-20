import torch
from torch import nn
from .builder import LOSSES
from torch.nn import MSELoss
from .builder import LOSSES
from torch.nn.functional import kl_div
@LOSSES.register_module()
class HFTLoss(LOSSES):
    def __init__(self, ada):
        self.ada = ada
        super(HFTLoss, self).__init__()

    def forward(self, pred, target, demonstration):
        return nn.MSELoss(pred, target) + self.ada * kl_div(pred, demonstration)