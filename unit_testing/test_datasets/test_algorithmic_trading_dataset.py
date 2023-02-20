import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.datasets.builder import build_dataset
from trademaster.datasets.algorithmic_trading import AlgorithmicTradingDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "algorithmic_trading", "dqn_btc.py"),
                        help="download datasets config file path")
    args = parser.parse_args()
    return args

def test_algorithmic_trading_dataset():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)

    dataset = build_dataset(cfg)

    assert isinstance(dataset, AlgorithmicTradingDataset)

if __name__ == '__main__':
    test_algorithmic_trading_dataset()