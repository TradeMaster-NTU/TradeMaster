import pandas as pd
from lib2to3.pgen2.pgen import DFAState
from logging import raiseExceptions
from xml.parsers.expat import model
from attr import define
from cv2 import dft
import ray
from ray.rllib.agents.a3c import a2c
from ray.rllib.agents.ddpg import ddpg
from ray.rllib.agents.pg import pg
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.sac import sac
from ray.rllib.agents.ddpg import td3
from env import TradingEnv
from pathlib import Path
import sys
sys.path.append('..')


class agent:
    """agent is class that contains API for the TradeMaster to train, valid and test the agents from the RLlib in the 
    finicial market
    to initialize the agent, we need:
    algorithms: the name of algorithms of RL, now, we support a2c,ddpg,pg,ppo,sac,td3
    train_config:the environment config for training.
    valid_config: the environment config for valid
    test_config:the environment config for test
    seed:the random seed we set for training, if the same will reproduce the same result
    num_epochs:the number of epochs the model is trained during the session
    env:the reinforcement learning environment for traning the model, need train_config,valid_config or test_config 
    to initialize
    model_path: the path to save and restore the trained model
    result_path: the path to save the result
    the overall process of RL is as following:
    we first use the data in the train config to train the agent for num_epochs epochs, at the end of each epoch, I stop and 
    use the trained agent to do trading in the valid environment and get the indicators of the trading process(sharpe ratio)
    in this case, then we use the epoch with the highest sharpe ratio in the valid environment to do trading in the test
    enviornment and get the result for that specific seed
    """

    def __init__(self, algorithms, train_config, seed, model_path, result_path, valid_config, test_config, train_env, test_env, num_epochs=10):
        self.model_path = model_path
        self.result_path = result_path
        self.algorithms = algorithms
        self.train_env = train_env
        self.test_env = test_env
        self.train_config = train_config
        self.valid_config = valid_config
        self.test_config = test_config
        self.num_epochs = num_epochs
        self.seed = seed

    def get_train_config(self):
        if self.algorithms == "a2c":
            model_config = a2c.A2C_DEFAULT_CONFIG.copy()
            trainer = a2c.A2CTrainer
        elif self.algorithms == "ddpg":
            model_config = ddpg.DEFAULT_CONFIG.copy()
            trainer = ddpg.DDPGTrainer
        elif self.algorithms == "pg":
            model_config = pg.DEFAULT_CONFIG.copy()
            trainer = pg.PGTrainer
        elif self.algorithms == "ppo":
            model_config = ppo.DEFAULT_CONFIG.copy()
            trainer = ppo.PPOTrainer
        elif self.algorithms == "sac":
            model_config = sac.DEFAULT_CONFIG.copy()
            trainer = sac.SACTrainer
        elif self.algorithms == "td3":
            model_config = td3.TD3_DEFAULT_CONFIG.copy()
            trainer = td3.TD3Trainer
        else:
            raiseExceptions("this algorithms has not been supported yet")
        self.model_config = model_config
        self.trainer = trainer
        return model_config, trainer

    def train_with_valid(self):
        model_config, trainer = self.get_train_config()
        self.sharpes = []
        self.checkpoints = []
        self.model_config["seed"] = self.seed
        self.model_config["env"] = self.train_env
        self.model_config["framework"] = "torch"
        self.model_config["env_config"] = self.train_config
        # so that we can reproduce our result
        self.model_config["num_workers"] = 1
        ray.init(ignore_reinit_error=True)
        self.trainer = self.trainer(
            env=self.train_env, config=self.model_config)
        for i in range(self.num_epochs):
            self.trainer.train()
            valid_env_instance = self.test_env(self.valid_config)
            state = valid_env_instance.reset()
            done = False
            while not done:
                action = self.trainer.compute_single_action(state)
                state, reward, done, sharpe = valid_env_instance.step(action)
            self.sharpes.append(sharpe)
            checkpoint = self.trainer.save()
            self.checkpoints.append(checkpoint)
        self.loc = self.sharpes.index(max(self.sharpes))
        self.trainer.restore(self.checkpoints[self.loc])
        print(self.model_path+"/"+str(self.seed)+"/")
        self.trainer.save(Path(self.model_path+"/"+str(self.seed)+"/"))
        new_path = self.model_path+"/"+str(self.seed)
        ray.shutdown()
        return new_path

    def test(self):

        test_env_instance = self.test_env(self.test_config)
        state = test_env_instance.reset()
        done = False
        while not done:
            action = self.trainer.compute_single_action(state)
            state, reward, done, sharpe = test_env_instance.step(action)
        rewards = test_env_instance.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = test_env_instance.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets

        self.result_path = self.result_path+"/"+str(self.seed)
        df.to_csv(self.result_path+".csv")
        return df
