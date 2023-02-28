from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_object, load_object,create_radar_score_baseline, calculate_radar_score, plot_radar_chart
import os
import ray
from ray.tune.registry import register_env
from trademaster.environments.portfolio_management.sarl_environment import PortfolioManagementSARLEnvironment
import pandas as pd
import numpy as np
import random
import torch


def env_creator(env_name):
    if env_name == 'portfolio_management_sarl':
        env = PortfolioManagementSARLEnvironment
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
        from ray.rllib.agents.ddpg.ddpg import TD3Trainer as trainer
    else:
        print(alg_name)
        print(alg_name == "A2C")
        print(type(alg_name))
        raise NotImplementedError
    return trainer

ray.init(ignore_reinit_error=True)
register_env("portfolio_management_sarl", lambda config: env_creator("portfolio_management_sarl")(config))

@TRAINERS.register_module()
class PortfolioManagementSARLTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementSARLTrainer, self).__init__()

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
        self.configs["env"] = PortfolioManagementSARLEnvironment
        self.configs["env_config"] = dict(dataset=self.dataset, task="train")

        self.init_before_training()

    def init_before_training(self):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            print(f"| Arguments Remove work_dir: {self.work_dir}")
        else:
            print(f"| Arguments Keep work_dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path, exist_ok=True)

    def train_and_valid(self):

        valid_score_list = []
        self.trainer = self.trainer_name(env="portfolio_management_sarl", config=self.configs)

        for epoch in range(1, self.epochs+1):
            print("Train Episode: [{}/{}]".format(epoch, self.epochs))
            self.trainer.train()

            config = dict(dataset=self.dataset, task="valid")
            self.valid_environment = env_creator("portfolio_management_sarl")(config)
            print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
            state = self.valid_environment.reset()

            episode_reward_sum = 0
            while True:
                action = self.trainer.compute_single_action(state)
                state, reward, done, information = self.valid_environment.step(action)
                episode_reward_sum += reward
                if done:
                    #print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break

            valid_score_list.append(information["sharpe_ratio"])

            checkpoint_path = os.path.join(self.checkpoints_path, "checkpoint-{:05d}.pkl".format(epoch))
            obj = self.trainer.save_to_object()
            save_object(obj, checkpoint_path)

        max_index = np.argmax(valid_score_list)
        obj = load_object(os.path.join(self.checkpoints_path, "checkpoint-{:05d}.pkl".format(max_index+1)))
        save_object(obj, os.path.join(self.checkpoints_path, "best.pkl"))
        ray.shutdown()

    def test(self):
        self.trainer = self.trainer_name(env="portfolio_management_sarl", config=self.configs)

        obj = load_object(os.path.join(self.checkpoints_path, "best.pkl"))
        self.trainer.restore_from_object(obj)

        config = dict(dataset=self.dataset, task="test")
        self.test_environment = env_creator("portfolio_management_sarl")(config)
        print("Test Best Episode")
        state = self.test_environment.reset()
        episode_reward_sum = 0
        while True:
            action = self.trainer.compute_single_action(state)
            state, reward, done, sharpe = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)

    def dynamics_test(self,test_dynamic,cfg):
        self.trainer = self.trainer_name(env="portfolio_management_sarl", config=self.configs)
        obj = load_object(os.path.join(self.checkpoints_path, "best.pkl"))
        self.trainer.restore_from_object(obj)

        test_dynamic_environments = []
        for i, path in enumerate(self.dataset.test_dynamic_paths):
            config = dict(dataset=self.dataset, task="test_dynamic",test_dynamic=test_dynamic,dynamics_test_path=path,task_index=i,work_dir=cfg.work_dir)
            test_dynamic_environments.append(env_creator("portfolio_management_sarl")(config))
        # for i,env in enumerate(test_dynamic_environments):
        #     state = env.reset()
        #     done = False
        #     while not done:
        #         action = self.trainer.compute_single_action(state)
        #         state, reward, done, sharpe = env.step(action)
        #     rewards = env.save_asset_memory()
        #     assets = rewards["total assets"].values
        #     df_return = env.save_portfolio_return_memory()
        #     daily_return = df_return.daily_return.values
        #     df = pd.DataFrame()
        #     df["daily_return"] = daily_return
        #     df["total assets"] = assets
        #     df.to_csv(os.path.join(self.work_dir, "test_dynamic_result"+"style_"+str(test_dynamic)+"_part_"+str(i)+".csv"), index=False)



        def Average_holding(states, env, weights_brandnew):
            if weights_brandnew is None:
                action = [0] + [1 / env.stock_dim for _ in range(env.stock_dim)]
                return action
            else:
                return weights_brandnew
        def Do_Nothing(states, env):
            return [1] + [0 for _ in range(env.stock_dim)]

        daily_return_list = []
        daily_return_list_Average_holding = []
        daily_return_list_Do_Nothing = []

        def test_single_env(this_env,policy,policy_id=None):
            this_env.test_id = policy_id
            state = this_env.reset()
            done = False
            weights_brandnew=None
            while not done:
                if policy_id=="Average_holding":
                    action = policy(state,this_env,weights_brandnew)
                elif policy_id=='Do_Nothing':
                    action = policy(state, this_env)
                else:
                    action = policy(state)
                # action = self.trainer.compute_single_action(state)
                state, reward, done, return_dict = this_env.step(action)
                if done:
                    break
                weights_brandnew = return_dict["weights_brandnew"]
            rewards = this_env.save_asset_memory()
            assets = rewards["total assets"].values
            df_return = this_env.save_portfolio_return_memory()
            daily_return = df_return.daily_return.values
            df = pd.DataFrame()
            df["daily_return"] = daily_return
            df["total assets"] = assets
            df.to_csv(os.path.join(self.work_dir, "test_dynamic_result"+"style_"+str(test_dynamic)+"_part_"+str(i)+".csv"), index=False)
            return daily_return

        for i,env in enumerate(test_dynamic_environments):
            #test agent
            daily_return_list.extend(test_single_env(env,self.trainer.compute_single_action,'agent'))
            #test Average_holding
            daily_return_list_Average_holding.extend(test_single_env(env,Average_holding,'Average_holding'))
            #test Do_Nothing
            daily_return_list_Do_Nothing.extend(test_single_env(env,Do_Nothing,'Do_Nothing'))
        metric_path = 'metric_' + str("test_dynamic") + '_' + str(
            cfg.data.test_dynamic)
        metrics_sigma_dict, zero_metrics = create_radar_score_baseline(cfg.work_dir, metric_path,
                                                                       zero_score_id='Do_Nothing',
                                                                       fifty_score_id='Average_holding')
        test_metrics_scores_dict = calculate_radar_score(cfg.work_dir, metric_path, 'agent', metrics_sigma_dict,
                                                         zero_metrics)
        radar_plot_path = cfg.work_dir
        # 'metric_' + str(self.task) + '_' + str(self.test_dynamic) + '_' + str(id) + '_radar.png')
        # print('test_metrics_scores are: ', test_metrics_scores_dict)
        # print('test_metrics_scores are:')
        # print_metrics(test_metrics_scores_dict)
        plot_radar_chart(test_metrics_scores_dict, 'radar_plot_agent_' + str(test_dynamic) + '.png',
                         radar_plot_path)
        # print('win rate is: ', sum(float(r) > 0 for r in daily_return_list) / len(daily_return_list))
        # print('Random_buy win rate is: ',
        #       sum(float(r) > 0 for r in daily_return_list_Average_holding) / len(daily_return_list_Average_holding))
        print("dynamics test end")