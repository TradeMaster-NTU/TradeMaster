import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.optimizers.builder import build_optimizer
from trademaster.optimizers import Adam
from trademaster.nets.builder import build_net
from trademaster.nets import QNet


def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "algorithmic_trading", "dqn_btc.py"),
                        help="download datasets config file path")
    args = parser.parse_args()
    return args

def test_adam():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)
    if args.verbose==1:
    print(cfg)

    act_net = build_net(cfg.act_net)
    cri_net = build_net(cfg.cri_net)
    assert isinstance(act_net, QNet)

    optimizer = build_optimizer(cfg, default_args=dict(params = act_net.parameters()))

    assert isinstance(optimizer, Adam)

if __name__ == '__main__':
    test_adam()