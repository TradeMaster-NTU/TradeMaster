from mmcv.utils import Registry
from trademaster.utils import build_from_cfg
import copy

IMPUTATION = Registry('imputation')

def build_imputation(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.data)
    imputation_model = build_from_cfg(cp_cfg, IMPUTATION, default_args)
    return imputation_model