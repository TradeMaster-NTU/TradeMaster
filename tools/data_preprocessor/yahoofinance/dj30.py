import os.path
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.preprocessor.builder import build_preprocessor

def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config",
                        default=osp.join(ROOT, "configs", "data_preprocessor", "yahoofinance", "dj30.py"),
                        help="download datasets config file path")
    args = parser.parse_args()
    return args


def preprocessor():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)

    preprocessor = build_preprocessor(cfg)

    preprocessor.run(os.path.join(ROOT, "data/portfolio_management/dj30/data.csv"))

if __name__ == '__main__':
    preprocessor()