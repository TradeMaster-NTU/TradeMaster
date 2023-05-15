import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.losses.builder import build_loss
from trademaster.losses import MSELoss


def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "algorithmic_trading", "dqn_btc.py"),
                        help="download datasets config file path")
    args = parser.parse_args()
    return args

def test_mse():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)
    if args.verbose==1:
    print(cfg)

    loss = build_loss(cfg)
    assert isinstance(loss, MSELoss)

if __name__ == '__main__':
    test_mse()