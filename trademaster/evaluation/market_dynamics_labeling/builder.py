from mmcv.utils import Registry
from trademaster.utils import build_from_cfg
import copy

Market_Dynamics_Model = Registry('market_dynamics_model')

def build_market_dynamics_model(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.market_dynamics_model)
    market_dynamics_model = build_from_cfg(cp_cfg, Market_Dynamics_Model, default_args)
    return market_dynamics_model