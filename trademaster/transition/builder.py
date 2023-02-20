from trademaster.utils import build_from_cfg
from mmcv.utils import Registry
import copy

TRANSITIONS = Registry('transition')

def build_transition(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.transition)
    transition = build_from_cfg(cp_cfg, TRANSITIONS, default_args)
    return transition