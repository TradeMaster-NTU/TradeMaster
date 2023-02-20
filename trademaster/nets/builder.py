import copy
from mmcv.utils import Registry
from trademaster.utils import build_from_cfg

NETS = Registry('net')

def build_net(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg)
    net = build_from_cfg(cp_cfg, NETS, default_args)
    return net
