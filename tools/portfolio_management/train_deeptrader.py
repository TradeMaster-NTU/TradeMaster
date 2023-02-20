import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import os
import torch

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

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
from trademaster.transition.builder import build_transition

def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "portfolio_management", "portfolio_management_dj30_deeptrader_deeptrader_adam_mse.py"),
                        help="download datasets config file path")
    parser.add_argument("--task_name", type=str, default="train")
    parser.add_argument("--test_style", type=str, default="-1")
    args = parser.parse_args()
    return args


def test_deeptrader():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    task_name = args.task_name

    cfg = replace_cfg_vals(cfg)
    # update test style
    cfg.data.update({'test_style': args.test_style})
    print(cfg)

    dataset = build_dataset(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="train"))
    valid_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="valid"))
    test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))
    if task_name.startswith("style_test"):
        test_style_environments = []
        for i, path in enumerate(dataset.test_style_paths):
            test_style_environments.append(build_environment(cfg, default_args=dict(dataset=dataset, task="test_style",
                                                                                    style_test_path=path,
                                                                                    task_index=i)))
    action_dim = train_environment.action_dim
    state_dim = train_environment.state_dim
    time_steps = train_environment.timesteps

    cfg.act.update(dict(N=action_dim, K_l=time_steps, num_inputs = state_dim))
    cfg.cri.update(dict(N=action_dim, K_l=time_steps, num_inputs = state_dim))
    cfg.market.update(dict(n_features = state_dim))

    act = build_net(cfg.act)
    cri = build_net(cfg.cri)
    market = build_net(cfg.market)

    work_dir = os.path.join(ROOT, cfg.trainer.work_dir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))
    cri_optimizer = build_optimizer(cfg, default_args=dict(params=cri.parameters()))
    market_optimizer = build_optimizer(cfg, default_args=dict(params=market.parameters()))

    criterion = build_loss(cfg)

    transition = build_transition(cfg)

    agent = build_agent(cfg, default_args=dict(action_dim=action_dim,
                                               state_dim=state_dim,
                                               act=act,
                                               cri=cri,
                                               market = market,
                                               act_optimizer=act_optimizer,
                                               cri_optimizer = cri_optimizer,
                                               market_optimizer = market_optimizer,
                                               criterion=criterion,
                                               transition=transition,
                                               device=device))

    if task_name.startswith("style_test"):
        trainers = []
        for env in test_style_environments:
            trainers.append(build_trainer(cfg, default_args=dict(train_environment=train_environment,
                                                                 valid_environment=valid_environment,
                                                                 test_environment=env,
                                                                 agent=agent,
                                                                 device=device)))
    else:
        trainer = build_trainer(cfg, default_args=dict(train_environment=train_environment,
                                                       valid_environment=valid_environment,
                                                       test_environment=test_environment,
                                                       agent=agent,
                                                       device=device,
                                                       ))
    if task_name.startswith("train"):
        trainer.train_and_valid()
        print("train end")
    elif task_name.startswith("test"):
        trainer.test()
        print("test end")
    elif task_name.startswith("style_test"):
        for trainer in trainers:
            trainer.test()
        print("style test end")


if __name__ == '__main__':
    test_deeptrader()
    """
    algorithmic_trading
    portfolio_management
    """