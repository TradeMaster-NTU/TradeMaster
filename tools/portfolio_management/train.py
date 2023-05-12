import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import os
import torch
import argparse
import os.path as osp
from mmcv import Config

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from trademaster.utils import replace_cfg_vals
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer
from trademaster.utils import set_seed

set_seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "portfolio_management", "portfolio_management_exchange_ppo_ppo_adam_mse.py"),
                        help="download datasets config file path")
    parser.add_argument("--task_name", type=str, default="train")
    parser.add_argument("--test_dynamic", type=str, default="-1")
    parser.add_argument("--verbose", type=int, default='1')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    task_name = args.task_name

    cfg = replace_cfg_vals(cfg)
    # update test style
    cfg.data.update({'test_dynamic': args.test_dynamic})
    if args.verbose == 1:
        print(cfg)

    dataset = build_dataset(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    work_dir = os.path.join(ROOT, cfg.trainer.work_dir)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    trainer = build_trainer(cfg, default_args=dict(dataset = dataset, device = device))

    if task_name.startswith("train"):
        trainer.train_and_valid()
        print("train end")
    elif task_name.startswith("test"):
        trainer.test()
        print("test end")

if __name__ == '__main__':
    main()
    """
    algorithmic_trading
    portfolio_management
    order_execution
    """