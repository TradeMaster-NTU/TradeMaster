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
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Download Alpaca Datasets")
    parser.add_argument(
        "--config",
        default=osp.join(
            ROOT,
            "configs",
            "high_frequency_trading",
            "high_frequency_trading_BTC_dqn_dqn_adam_mse.py",
        ),
        help="download datasets config file path",
    )
    parser.add_argument("--task_name", type=str, default="train")
    parser.add_argument("--test_style", type=str, default="-1")
    args = parser.parse_args()
    return args


def test_dqn():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    task_name = args.task_name

    cfg = replace_cfg_vals(cfg)
    # update test style
    cfg.data.update({"test_style": args.test_style})
    print(cfg)

    dataset = build_dataset(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="valid")
    )
    test_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="test")
    )
    cfg.environment = cfg.train_enviroment
    train_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="train")
    )

    action_dim = train_environment.action_dim
    state_dim = train_environment.state_dim

    cfg.act.update(dict(action_dim=action_dim, state_dim=state_dim))
    act = build_net(cfg.act)
    act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))
    if cfg.cri:
        cfg.cri.update(dict(action_dim=action_dim, state_dim=state_dim))
        cri = build_net(cfg.cri)
        cri_optimizer = build_optimizer(cfg, default_args=dict(params=cri.parameters()))
    else:
        cri = None
        cri_optimizer = None

    criterion = build_loss(cfg)

    agent = build_agent(
        cfg,
        default_args=dict(
            action_dim=action_dim,
            state_dim=state_dim,
            act=act,
            cri=cri,
            act_optimizer=act_optimizer,
            cri_optimizer=cri_optimizer,
            criterion=criterion,
            device=device,
        ),
    )

    trainer = build_trainer(
        cfg,
        default_args=dict(
            train_environment=train_environment,
            valid_environment=valid_environment,
            test_environment=test_environment,
            agent=agent,
            device=device,
        ),
    )

    cfg.dump(osp.join(ROOT, cfg.work_dir, osp.basename(args.config)))

    if task_name.startswith("train"):
        trainer.train_and_valid()
        trainer.test()
        print("train end")
    elif task_name.startswith("test"):
        trainer.test()
        print("test end")


if __name__ == "__main__":
    test_dqn()
