from mmcv.utils import Registry
from trademaster.utils import build_from_cfg
import copy

COLLECTORS = Registry('collector')

def build_collector(cfg, default_args = None):
    cp_cfg = copy.deepcopy(cfg.collector)
    collector = build_from_cfg(cp_cfg, COLLECTORS, default_args)
    return collector