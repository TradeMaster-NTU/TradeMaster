from mmcv.utils import Registry
from trademaster.utils import build_from_cfg
import copy

PREPROCESSOR = Registry('preprocessor')

def build_preprocessor(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.data)
    dataset = build_from_cfg(cp_cfg, PREPROCESSOR, default_args)
    return dataset