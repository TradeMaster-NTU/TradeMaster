import sys

sys.path.append(".")
import torch
import argparse
import yaml
from ray.tune.registry import register_env
import ray
import pandas as pd
import numpy as np
import os
import random
from env.PM.portfolio_for_SARL import TradingEnv as env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--env_name",
    choices=["portfolio"],
    default="portfolio",
    help="the name of TradingEnv ",
)
parser.add_argument(
    "--dict_trained_model",
    default="result/SARL/trained_model/",
    help="the dict of the trained model ",
)

parser.add_argument(
    "--train_env_config_dict",
    default="config/input_config/env/portfolio/portfolio_for_SARL/train.yml",
    help="the dict of the train config of TradingEnv ",
)

parser.add_argument(
    "--valid_env_config_dict",
    default="config/input_config/env/portfolio/portfolio_for_SARL/valid.yml",
    help="the dict of the valid config of TradingEnv ",
)

parser.add_argument(
    "--test_env_config_dict",
    default="config/input_config/env/portfolio/portfolio_for_SARL/test.yml",
    help="the dict of the test config of TradingEnv ",
)

parser.add_argument(
    "--name_of_algorithms",
    choices=["PPO", "A2C", "SAC", "TD3", "PG", "DDPG"],
    type=str,
    default="DDPG",
    help="name_of_algorithms ",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="the number of training epoch",
)

parser.add_argument(
    "--random_seed",
    type=int,
    default=12345,
    help="the number of training epoch",
)

parser.add_argument(
    "--model_config_dict",
    type=str,
    default="config/input_config/agent/SOTA/DDPG.yml",
    help="the dict of the model_config file",
)

parser.add_argument(
    "--result_dict",
    type=str,
    default="result/SARL/test_result/",
    help="the dict of the result of the test",
)


def env_creator(env_name):
    if env_name == 'portfolio':
        from env.PM.portfolio_for_SARL import TradingEnv as env

    else:
        raise NotImplementedError
    return env


def load_yaml(dict):
    realpath = os.path.abspath(".")
    file_dict = os.path.join(realpath, dict)
    with open(file_dict, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def select_algorithms(alg_name):
    if alg_name == "A2C":
        from ray.rllib.agents.a3c.a2c import A2CTrainer as trainer
    elif alg_name == "DDPG":
        from ray.rllib.agents.ddpg.ddpg import DDPGTrainer as trainer
    elif alg_name == 'PG':
        from ray.rllib.agents.pg.pg import PGTrainer as trainer
    elif alg_name == 'PPO':
        from ray.rllib.agents.ppo.ppo import PPOTrainer as trainer
    elif alg_name == 'SAC':
        from ray.rllib.agents.sac.sac import SACTrainer as trainer
    elif alg_name == 'TD3':
        from ray.rllib.agents.ddpg.td3 import TD3Trainer as trainer
    else:
        print(alg_name)

        raise NotImplementedError
    return trainer


# register_env('Train', lambda config: env_creator("portfolio")(config))


class agent:
    def __init__(self, args):
        self.trained_model_dict = args.dict_trained_model
        self.num_epochs = args.num_epochs
        self.env_name = args.env_name
        self.seed = args.random_seed
        self.train_env_config_dict = args.train_env_config_dict
        self.valid_env_config_dict = args.valid_env_config_dict
        self.test_env_config_dict = args.test_env_config_dict
        self.name_of_algorithms = args.name_of_algorithms
        self.result_dict = args.result_dict
        select_algorithms(self.name_of_algorithms)
        self.model_config_dict = args.model_config_dict
        # self.model_config = load_yaml(self.model_config_dict)
        from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG
        self.model_config = DEFAULT_CONFIG
        self.train_env_config = load_yaml(self.train_env_config_dict)
        self.valid_env_config = load_yaml(self.valid_env_config_dict)
        self.test_env_config = load_yaml(self.test_env_config_dict)
        self.setseed()
        ray.init(ignore_reinit_error=True)
        self.trainer = select_algorithms(self.name_of_algorithms)
        register_env(self.env_name, lambda config: env_creator(self.env_name)
                     (config))
        self.model_config["env"] = env
        self.model_config["env_config"] = self.train_env_config

    def setseed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        tf.random.set_seed(self.seed)
        torch.manual_seed(self.seed)

    def train_with_valid(self):
        #TODO there is None in the checkpiont which is strange(It has been solved if your python downgraded to 3.7.13)
        self.sharpes = []
        self.checkpoints = []

        self.trainer = self.trainer(env=self.env_name,
                                    config=self.model_config)

        for i in range(self.num_epochs):

            self.trainer.train()
            valid_env_instance = env_creator(self.env_name)(
                self.valid_env_config)
            state = valid_env_instance.reset()
            done = False
            while not done:
                action = self.trainer.compute_single_action(state)
                state, reward, done, information = valid_env_instance.step(
                    action)
            self.sharpes.append(information["sharpe_ratio"])
            checkpoint = self.trainer.save()
            self.checkpoints.append(checkpoint)
        self.loc = self.sharpes.index(max(self.sharpes))
        self.trainer.restore(self.checkpoints[self.loc])
        self.trainer.save(self.trained_model_dict)
        ray.shutdown()

    def test(self):
        self.test_env_instance = env_creator(self.env_name)(
            self.test_env_config)
        state = self.test_env_instance.reset()
        done = False
        while not done:
            action = self.trainer.compute_single_action(state)
            state, reward, done, sharpe = self.test_env_instance.step(action)
        rewards = self.test_env_instance.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_env_instance.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        if not os.path.exists(self.result_dict):
            os.makedirs(self.result_dict)
        df.to_csv(self.result_dict + "result.csv")


if __name__ == "__main__":
    args = parser.parse_args()

    a = agent(args)
    a.train_with_valid()
    a.test()