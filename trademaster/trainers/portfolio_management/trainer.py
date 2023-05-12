import torch
import numpy as np
import pandas as pd
from trademaster.environments.portfolio_management.environment import PortfolioManagementEnvironment
from ray.tune.registry import register_env
import ray
import os
from trademaster.utils import get_attr, save_object, load_object,plot_metric_against_baseline
from ..builder import TRAINERS
from ..custom import Trainer
import random
import shutil
from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[3]


def env_creator(env_name):
    if env_name == 'portfolio_management':
        env = PortfolioManagementEnvironment
    else:
        raise NotImplementedError
    return env


def select_algorithms(alg_name):
    alg_name = alg_name.upper()
    if alg_name == "A2C":
        from ray.rllib.agents.a3c.a2c import A2CTrainer as trainer
    elif alg_name == "DDPG":
        from ray.rllib.agents.ddpg.ddpg import DDPGTrainer as trainer
    elif alg_name == 'PG':
        from ray.rllib.agents.pg import PGTrainer as trainer
    elif alg_name == 'PPO':
        from ray.rllib.agents.ppo.ppo import PPOTrainer as trainer
    elif alg_name == 'SAC':
        from ray.rllib.agents.sac import SACTrainer as trainer
    elif alg_name == 'TD3':
        from ray.rllib.agents.ddpg.td3 import TD3Trainer as trainer
    else:
        print(alg_name)
        print(alg_name == "A2C")
        print(type(alg_name))
        raise NotImplementedError
    return trainer

# os.environ["RAY_LOG_TO_STDERR"] = "1"
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.disable(logging.INFO)
logging.disable(logging.WARNING)
# handler = logging.StreamHandler(sys.stdout)
ray.init(ignore_reinit_error=True)
register_env("portfolio_management", lambda config: env_creator(
    "portfolio_management")(config))

@ray.remote
class Actor:
    def __init__(self):
        # Basic config automatically configures logs to
        # be streamed to stdout and stderr.
        # Set the severity to INFO so that info logs are printed to stdout.
        logging.basicConfig(level=logging.DEBUG)

    def log(self, msg):
        logging.info(msg)

@ray.remote
def f(msg):
    logging.basicConfig(format='%(message)s',level=logging.DEBUG)
    logging.info(msg)

# ray.get(f.remote("A log message for a task"))

@TRAINERS.register_module()
class PortfolioManagementTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementTrainer, self).__init__()

        self.device = get_attr(kwargs, "device", None)

        self.configs = get_attr(kwargs, "configs", None)
        self.agent_name = get_attr(kwargs, "agent_name", "ppo")
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.dataset = get_attr(kwargs, "dataset", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)
        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.num_threads = int(get_attr(kwargs, "num_threads", 8))

        self.trainer_name = select_algorithms(self.agent_name)
        self.configs["env"] = "portfolio_management"
        self.configs["env_config"] = dict(dataset=self.dataset, task="train")
        self.verbose = get_attr(kwargs, "verbose", False)
        self.init_before_training()

    def init_before_training(self):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(
                input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            if self.verbose:
                print(f"| Arguments Remove work_dir: {self.work_dir}")
        else:
            if self.verbose:
                print(f"| Arguments Keep work_dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path, exist_ok=True)

    def train_and_valid(self):

        valid_score_list = []
        save_dict_list = []
        self.trainer = self.trainer_name(
            env="portfolio_management", config=self.configs)

        for epoch in range(1, self.epochs + 1):
            ray.get(f.remote("Train Episode: [{}/{}]".format(epoch, self.epochs)))
            # ray.get(self.Actor.log.remote("Train Episode: [{}/{}]".format(epoch, self.epochs)))
            self.trainer.train()
            config = dict(dataset=self.dataset, task="valid")
            self.valid_environment = env_creator(
                "portfolio_management")(config)
            ray.get(f.remote("Valid Episode: [{}/{}]".format(epoch, self.epochs)))
            state = self.valid_environment.reset()

            episode_reward_sum = 0
            while True:
                action = self.trainer.compute_single_action(state)
                action = np.exp(action)/np.sum(np.exp(action))
                state, reward, done, information = self.valid_environment.step(
                    action)
                episode_reward_sum += reward
                if done:
                    #ray.get(f.remote("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break
            ray.get(f.remote(information['table']))
            save_dict_list.append(information)
            valid_score_list.append(information["sharpe_ratio"])

            checkpoint_path = os.path.join(
                self.checkpoints_path, "checkpoint-{:05d}.pkl".format(epoch))
            obj = self.trainer.save_to_object()
            save_object(obj, checkpoint_path)

        max_index = np.argmax(valid_score_list)
        plot_metric_against_baseline(total_asset=save_dict_list[max_index]['total_assets'],buy_and_hold=None,alg=self.agent_name.upper(),task='valid',color='darkcyan',save_dir=self.work_dir)

        obj = load_object(os.path.join(self.checkpoints_path,
                          "checkpoint-{:05d}.pkl".format(max_index+1)))
        save_object(obj, os.path.join(self.checkpoints_path, "best.pkl"))
        ray.shutdown()

    def test(self):
        self.trainer = self.trainer_name(
            env="portfolio_management", config=self.configs)

        obj = load_object(os.path.join(self.checkpoints_path, "best.pkl"))
        self.trainer.restore_from_object(obj)

        config = dict(dataset=self.dataset, task="test")
        self.test_environment = env_creator("portfolio_management")(config)
        ray.get(f.remote("Test Best Episode"))
        state = self.test_environment.reset()
        episode_reward_sum = 0
        while True:
            action = self.trainer.compute_single_action(state)
            action = np.exp(action)/np.sum(np.exp(action))

            state, reward, done, sharpe = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                plot_metric_against_baseline(total_asset=sharpe['total_assets'], buy_and_hold=None,
                                             alg=self.agent_name.upper(), task='test', color='darkcyan', save_dir=self.work_dir)

                # ray.get(f.remote("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break
        ray.get(f.remote(sharpe['table']))
        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
        return daily_return
