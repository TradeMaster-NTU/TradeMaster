from trademaster.utils import build_from_cfg
from mmcv.utils import Registry
import copy

OPTIMIZERS = Registry('optimizer')

def build_optimizer(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.optimizer)
    optimizer = build_from_cfg(cp_cfg, OPTIMIZERS, default_args)
    return optimizer