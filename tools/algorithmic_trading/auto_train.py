import sys
from pathlib import Path
import os

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
from trademaster.transition.builder import build_transition
import pdb
import optuna
from shutil import copyfile
import joblib

from trademaster.utils import set_seed
set_seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "algorithmic_trading", "algorithmic_trading_BTC_dqn_dqn_adam_mse.py"),
                        help="download datasets config file path")
    parser.add_argument("--task_name", type=str, default="train")
    parser.add_argument("--test_style", type=str, default='-1')
    parser.add_argument("--auto_tuning", default=False, type=bool)
    parser.add_argument("--n_trials", default=10, type=int)
    args = parser.parse_args()
    return args

def test_dqn_builder(args, sampled_params):
    cfg = Config.fromfile(args.config)
    task_name = args.task_name
    cfg = replace_cfg_vals(cfg)
    cfg.data.update({'test_style': args.test_style})
    
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

    cfg.act.update(dict(action_dim=action_dim, state_dim=state_dim))

    if args.auto_tuning == True:
        lr = sampled_params['lr']
        explore_rate = sampled_params['explore_rate']
        dims = sampled_params['dims']
        cfg.optimizer.update(dict(lr=lr))
        cfg.act.update(dict(explore_rate=explore_rate, dims = dims))

    print(cfg)

    act = build_net(cfg.act)
    act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))

    if cfg.cri:
        cfg.cri.update(dict(action_dim=action_dim, state_dim=state_dim))
        cri_net = build_net(cfg.cri)
        cri_optimizer = build_optimizer(cfg, default_args=dict(params=cri.parameters()))
    else:
        cri = None
        cri_optimizer = None

    if_remove = cfg.trainer.if_remove
    work_dir = os.path.join(ROOT, cfg.work_dir)
    

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    cfg.dump(osp.join(ROOT, cfg.work_dir, osp.basename(args.config)))


    criterion = build_loss(cfg)

    transition = build_transition(cfg)

    agent = build_agent(cfg, default_args=dict(action_dim = action_dim,
                                               state_dim = state_dim,
                                               act = act,
                                               cri = cri,
                                               act_optimizer = act_optimizer,
                                               cri_optimizer = cri_optimizer,
                                               criterion = criterion,
                                               transition = transition,
                                               device = device))

    if task_name.startswith("style_test"):
        trainers = []
        for env in test_style_environments:
            trainers.append(build_trainer(cfg, default_args=dict(train_environment=train_environment,
                                                                 valid_environment=valid_environment,
                                                                 test_environment=env,
                                                                 agent=agent,
                                                                 device=device)))
    else:
        trainer = build_trainer(cfg, default_args=dict(train_environment = train_environment,
                                                   valid_environment = valid_environment,
                                                   test_environment = test_environment,
                                                   agent = agent,
                                                   device = device))

    if task_name.startswith("style_test"):
        return trainers
    else:
        return trainer


def sample_params(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    explore_rate = trial.suggest_float("explore_rate", 0.15, 0.3, step=0.05)
    dims = trial.suggest_categorical("hidden_nodes", [(64,32), (128,64)])
    trial_number = trial.number
    sampled_params = dict(lr=lr, explore_rate=explore_rate, dims=dims, trial_number=trial_number)
    return sampled_params


def objective(trial: optuna.Trial) -> float:
    args = parse_args()
    sampled_params = sample_params(trial)
    trainer = test_dqn_builder(args, sampled_params)
    valid_score = trainer.train_and_valid_trial(sampled_params['trial_number'])
    return valid_score

def test_dqn():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.auto_tuning == False:
        if args.task_name.startswith("style_test"):
            trainers = test_dqn_builder(args, sampled_params=None)
        else:
            trainer = test_dqn_builder(args, sampled_params=None)
        if args.task_name.startswith("train"):
            trainer.train_and_valid()
            trainer.test()
            print("train end")
        elif args.task_name.startswith("test"):
            trainer.test()
            print("test end")
        elif args.task_name.startswith("style_test"):
            daily_return_list = []
            for trainer in trainers:
                daily_return_list.extend(trainer.test())
            print('win rate is: ', sum(r > 0 for r in daily_return_list) / len(daily_return_list))
            print("style test end")
    else:
        if args.task_name.startswith("train"):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.n_trials)
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            print("  Best trial number: ", trial.number)
            best_model_path = os.path.join(ROOT, cfg.work_dir, "checkpoints", "trial-{:05d}.pth".format(trial.number))
            copyfile(best_model_path, os.path.join(ROOT, cfg.work_dir, "checkpoints", "best.pth"))
            joblib.dump(study, os.path.join(ROOT, cfg.work_dir, "auto_tuning.pkl"))
            trainer = test_dqn_builder(args, sampled_params = sample_params(study.best_trial))
            trainer.test()
            print("train end")
        elif args.task_name.startswith("test"):
            study = joblib.load(os.path.join(ROOT, cfg.work_dir, "auto_tuning.pkl"))
            trainer = test_dqn_builder(args, sampled_params = sample_params(study.best_trial))
            trainer.test()
            print("test end")
    


if __name__ == '__main__':
    test_dqn()
