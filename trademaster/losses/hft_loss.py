import torch
from torch import nn
from .builder import LOSSES
from torch.nn import MSELoss
from .builder import LOSSES
from torch.nn.functional import kl_div
from .custom import Loss


@LOSSES.register_module()
class HFTLoss(Loss):
    def __init__(self, ada):
        self.ada = ada
        super(HFTLoss, self).__init__()

    def forward(self, pred, target, distribution, demonstration):
        return nn.MSELoss()(pred, target) + self.ada * kl_div(
            (distribution.softmax(dim=-1) + 1e-8).log(),
            demonstration,
            reduction="batchmean",
        )
