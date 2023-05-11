import sys
from pathlib import Path
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
from trademaster.agents.algorithmic_trading import AlgorithmicTradingDQN


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

    act_net = build_net(cfg.act_net)
    cri_net = build_net(cfg.cri_net)

    optimizer = build_optimizer(cfg, default_args=dict(params=act_net.parameters()))
    loss = build_loss(cfg)

    dataset = build_dataset(cfg)

    train_environment = build_environment(cfg, default_args=dict(dataset = dataset, task = "train"))

    agent = build_agent(cfg, default_args=dict(n_action=train_environment.action_space.n,
                                               n_states = train_environment.observation_space.shape[0],
                                               act_net=act_net,
                                               cri_net=cri_net,
                                               optimizer=optimizer,
                                               loss=loss))
    assert isinstance(agent, AlgorithmicTradingDQN)

if __name__ == '__main__':
    test_dqn()