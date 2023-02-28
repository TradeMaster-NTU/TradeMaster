import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

import torch
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.imputation.builder import build_imputation


def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "missing_value_imputation", "missing_value_imputation.py"),
                        help="download datasets config file path")
    parser.add_argument("--dataset", default="dj30", help="dataset name")
    parser.add_argument("--tic", default="IBM", help="ticker name")                     
    args = parser.parse_args()
    return args

def imputation():
    args = parse_args()
    dataset_name = args.dataset
    tic_name = args.tic

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)
    cfg.data.update(dict(dataset_name=dataset_name, tic_name=tic_name))
    print(cfg)


    imputation = build_imputation(cfg)

    imputation.run()

if __name__ == '__main__':
    imputation()