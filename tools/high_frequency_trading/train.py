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
from trademaster.utils import replace_cfg_vals ,create_radar_score_baseline, calculate_radar_score, plot_radar_chart
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
    parser.add_argument("--test_dynamic", type=str, default="-1")
    args = parser.parse_args()
    return args


def test_dqn():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    task_name = args.task_name
    test_dynamic = args.test_dynamic

    cfg = replace_cfg_vals(cfg)
    # update test style
    cfg.data.update({"test_dynamic": args.test_dynamic})
    print(cfg)

    dataset = build_dataset(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="valid")
    )
    test_environment = build_environment(
        cfg, default_args=dict(dataset=dataset, task="test")
    )
    if task_name.startswith("dynamics_test"):
        test_dynamic_environments = []
        for i, path in enumerate(dataset.test_dynamic_paths):
            test_dynamic_environments.append(
                build_environment(cfg, default_args=dict(dataset=dataset, task="test_dynamic",
                                                         dynamics_test_path=path,
                                                         test_dynamic=test_dynamic,
                                                         task_index=i, work_dir=cfg.work_dir)))


    cfg.environment = cfg.train_environment
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
    if task_name.startswith("dynamics_test"):
        trainers = []
        for env in test_dynamic_environments:
            trainers.append(build_trainer(cfg, default_args=dict(train_environment=train_environment,
                                                                 valid_environment=valid_environment,
                                                                 test_environment=env,
                                                                 agent=agent,
                                                                 device=device)))
    else:
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
    elif task_name.startswith("dynamics_test"):
        def Blind_Bid(states,env):
            return env.action_dim -1
        def Do_Nothing(states,env):
            return 0
        daily_return_list = []
        daily_return_list_Blind_Bid=[]
        daily_return_list_Do_Nothing=[]
        for trainer in trainers:
            daily_return_list.extend(trainer.test())
            daily_return_list_Blind_Bid.extend(trainer.test_with_customize_policy(Blind_Bid,'Blind_Bid'))
            daily_return_list_Do_Nothing.extend(trainer.test_with_customize_policy(Do_Nothing,'Do_Nothing'))
            metric_path='metric_' + str(trainer.test_environment.task) + '_' + str(trainer.test_environment.test_dynamic)
        metrics_sigma_dict,zero_metrics=create_radar_score_baseline(cfg.work_dir,metric_path)
        test_metrics_scores_dict = calculate_radar_score(cfg.work_dir,metric_path,'agent',metrics_sigma_dict,zero_metrics)
        radar_plot_path=cfg.work_dir
        # 'metric_' + str(self.task) + '_' + str(self.test_dynamic) + '_' + str(id) + '_radar.png')
        # print('test_metrics_scores are: ',test_metrics_scores_dict)
        # print('test_metrics_scores are:')
        # print_metrics(test_metrics_scores_dict)
        plot_radar_chart(test_metrics_scores_dict,'radar_plot_agent_'+str(test_dynamic)+'.png',radar_plot_path)
        # print('win rate is: ', sum(float(r) > 0 for r in daily_return_list) / len(daily_return_list))
        # print('blind_bid win rate is: ', sum(float(r) > 0 for r in daily_return_list_Blind_Bid) / len(daily_return_list_Blind_Bid))
        print("dynamics test end")


if __name__ == "__main__":
    test_dqn()
