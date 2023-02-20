from mmcv.utils import Registry
from trademaster.utils import build_from_cfg
import copy

ENVIRONMENTS = Registry('environment')

def build_environment(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.environment)
    environment = build_from_cfg(cp_cfg, ENVIRONMENTS, default_args)
    return environment