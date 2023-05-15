import sys
from pathlib import Path
import os

ROOT = str(Path(__file__).resolve().parents[3])
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


def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "algorithmic_trading", "dqn_btc.py"),
                        help="download datasets config file path")
    args = parser.parse_args()
    return args


def test_dqn():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)
    if args.verbose==1:
    print(cfg)

    dataset = build_dataset(cfg)

    train_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="train"))
    valid_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="valid"))
    test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))

    n_action = train_environment.action_space.n
    n_state = train_environment.observation_space.shape[0]

    cfg.act_net.update(dict(n_action=n_action, n_state=n_state))
    cfg.cri_net.update(dict(n_action=n_action, n_state=n_state))

    act_net = build_net(cfg.act_net)
    cri_net = build_net(cfg.cri_net)

    if_remove = cfg.trainer.if_remove
    work_dir = os.path.join(ROOT, cfg.trainer.work_dir)
    if if_remove is None:
        if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {work_dir}? ") == 'y')
    elif if_remove:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"| Arguments Remove cwd: {work_dir}")
    else:
        print(f"| Arguments Keep cwd: {work_dir}")

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    optimizer = build_optimizer(cfg, default_args=dict(params=act_net.parameters()))
    loss = build_loss(cfg)

    agent = build_agent(cfg, default_args=dict(n_action=n_action,
                                               n_state=n_state,
                                               act_net=act_net,
                                               cri_net=cri_net,
                                               optimizer=optimizer,
                                               loss=loss))

    trainer = build_trainer(cfg, default_args=dict(train_environment=train_environment,
                                                   valid_environment=valid_environment,
                                                   test_environment=test_environment,
                                                   agent=agent))
    trainer.train_and_valid()


if __name__ == '__main__':
    test_dqn()
