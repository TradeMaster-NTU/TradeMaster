import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import os
import torch
import argparse
import os.path as osp
from mmcv import Config
from collections import Counter
from collections import OrderedDict
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from trademaster.utils import replace_cfg_vals,print_metrics
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer
from trademaster.transition.builder import build_transition
from trademaster.utils import set_seed

set_seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "order_execution",
                                                     "order_execution_BTC_eteo_eteo_adam_mse.py"),
                        help="download datasets config file path")
    parser.add_argument("--task_name", type=str, default="test")
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

    train_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="train"))
    valid_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="valid"))
    test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))
    if task_name.startswith("dynamics_test"):
        test_dynamic_environments = []
        for i, path in enumerate(dataset.test_dynamic_paths):
            test_dynamic_environments.append(build_environment(cfg, default_args=dict(dataset=dataset, task="test_dynamic",
                                                                                    dynamics_test_path=path,
                                                                                    task_index=i)))

    action_dim = train_environment.action_dim
    state_dim = train_environment.state_dim

    cfg.act.update(dict(action_dim=action_dim, state_dim=state_dim))
    cfg.cri.update(dict(action_dim=action_dim, state_dim=state_dim))

    act = build_net(cfg.act)
    cri = build_net(cfg.cri)

    work_dir = os.path.join(ROOT, cfg.trainer.work_dir)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))
    cri_optimizer = None

    criterion = build_loss(cfg)
    transition = build_transition(cfg)

    agent = build_agent(cfg, default_args=dict(action_dim=action_dim,
                                               state_dim=state_dim,
                                               act=act,
                                               cri=cri,
                                               act_optimizer=act_optimizer,
                                               cri_optimizer=cri_optimizer,
                                               criterion=criterion,
                                               transition=transition,
                                               device=device))
    if task_name.startswith("dynamics_test"):
        trainers = []
        for env in test_dynamic_environments:
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
        trainer.test()
        print("train end")
    elif task_name.startswith("test"):
        trainer.test()
        print("test end")
    elif task_name.startswith("dynamics_test"):
        return_list= []
        for trainer in trainers:
            return_list.append(trainer.test())
        cash_left_list=[]
        TWAP_value_list=[]
        for r in return_list:
            cash_left_list.append(r['cash_left'])
            TWAP_value_list.append(r['TWAP_value'])
        cash_left_mean=sum(cash_left_list)/len(cash_left_list)
        TWAP_value_mean=sum(TWAP_value_list)/len(TWAP_value_list)
        stats = OrderedDict(
            {
                "Cash Left": ["{:04f}".format(cash_left_mean)],
                "TWAP": ["{:04f}".format(TWAP_value_mean)],
                "Cash Left Ratio": ["{:04f}%".format(100 * (cash_left_mean - TWAP_value_mean) / TWAP_value_mean)],
            }
        )
        table = print_metrics(stats)
        print('Summary of dynamics test:')
        print(table)
        # print('The win rate of this dynamic is:')
        # print(Counter(reward_list))
        print("dynamics test end")


if __name__ == '__main__':
    main()
    """
    algorithmic_trading
    portfolio_management
    """
