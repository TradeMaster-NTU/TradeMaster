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
from trademaster.utils import replace_cfg_vals,create_radar_score_baseline, calculate_radar_score, plot_radar_chart
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer
from trademaster.evaluation.market_dynamics_labeling.builder import build_market_dynamics_model

def parse_args():
    parser = argparse.ArgumentParser(description='Market_Dynamics_modeling args')

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--fitting_parameters",nargs='+', type=str)
    parser.add_argument("--labeling_parameters",  nargs="+", type=float)
    parser.add_argument('--regime_number',type=int,default=4)
    parser.add_argument('--length_limit',type=int,default=0)
    parser.add_argument('--OE_BTC',type=bool,default=False)
    parser.add_argument('--PM',type=str,default='False')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "evaluation", "market_dynamics_modeling.py"),
                        help="deafult mdm config path")

    args = parser.parse_args()
    return args


def run_mdm():
    args = parse_args()

    cfg = Config.fromfile(args.config)


    # task_name = args.task_name
    # test_dynamic=args.test_dynamic

    cfg = replace_cfg_vals(cfg)
    # update test style

    model = build_market_dynamics_model(cfg)

    process_datafile_path, market_dynamic_labeling_visualization_paths=model.run()
    print(f'the processed datafile is at {process_datafile_path}')
    plot_dir = os.path.dirname(os.path.realpath(market_dynamic_labeling_visualization_paths[0]))
    print(f'the visualizations are at {plot_dir}')

    ## wirte path to cfg
    cfg.market_dynamics_model.process_datafile_path=process_datafile_path.replace("\\", "/")
    cfg.market_dynamics_model.market_dynamic_labeling_visualization_paths=market_dynamic_labeling_visualization_paths
    cfg.dump(args.config)



if __name__ == '__main__':
    run_mdm()
