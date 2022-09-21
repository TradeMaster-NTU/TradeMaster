from logging import raiseExceptions
from stat import S_ENFMT
import torch.nn as nn
import pandas as pd
import sys

sys.path.append(".")
# from EIIE.model import EIIE_critirc
from agent.EIIE.model import EIIE_con, EIIE_lstm, EIIE_rnn, EIIE_critirc
import argparse
from agent.EIIE.util import *
from env.PM.portfolio_for_EIIE import Tradingenv

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed",
                    type=int,
                    default=12345,
                    help="the path for storing the downloaded data")
parser.add_argument(
    "--env_config_path",
    type=str,
    default="config/input_config/env/portfolio/portfolio_for_EIIE/",
    help="the path for storing the downloaded data")
parser.add_argument(
    "--net_type",
    choices=["conv", "lstm", "rnn"],
    default="conv",
    help="the name of the model",
)
parser.add_argument(
    "--num_hidden_nodes",
    type=int,
    default=32,
    help="the number of hidden nodes in lstm or rnn",
)
parser.add_argument(
    "--num_out_channel",
    type=int,
    default=2,
    help="the number of channel",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="the gamma for DPG",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="result/EIIE/trained_model",
    help="the path for trained model",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/EIIE/test_result",
    help="the path for test result",
)
parser.add_argument(
    "--num_epoch",
    type=int,
    default=10,
    help="the number of epoch we train",
)


class trader:
    def __init__(self, args):
        self.num_epoch = args.num_epoch
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cpu' if self.GPU_IN_USE else 'cpu')
        self.model_path = args.model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.result_path = args.result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.train_env_config = load_yaml(args.env_config_path + "train.yml")
        self.valid_env_config = load_yaml(args.env_config_path + "valid.yml")
        self.test_env_config = load_yaml(args.env_config_path + "test.yml")
        self.train_env_instance = Tradingenv(self.train_env_config)
        self.valid_env_instance = Tradingenv(self.valid_env_config)
        self.test_env_instance = Tradingenv(self.test_env_config)
        self.day_length = self.train_env_config["length_day"]
        self.input_channel = len(self.train_env_config["tech_indicator_list"])
        if args.net_type == "conv":
            from agent.EIIE.model import EIIE_con as net
            self.net = net(self.input_channel, args.num_out_channel,
                           self.day_length)
        elif args.net_type == "lstm":
            from agent.EIIE.model import EIIE_lstm as net
            self.net = net(self.input_channel, 1, args.num_hidden_nodes)
        elif args.net_type == "rnn":
            from agent.EIIE.model import EIIE_rnn as net
            self.net = net(self.input_channel, 1, args.num_hidden_nodes)
        else:
            raiseExceptions("this kind of nets are not implemented yet")
        self.critic = EIIE_critirc(self.input_channel, 1,
                                   args.num_hidden_nodes)
        self.test_action_memory = []  # to store the
        self.optimizer_actor = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),
                                                 lr=1e-4)
        self.memory_counter = 0
        self.memory_capacity = 1000
        self.s_memory = []
        self.a_memory = []
        self.r_memory = []
        self.sn_memory = []
        self.policy_update_frequency = 500
        self.critic_learn_time = 0
        self.gamma = args.gamma
        self.mse_loss = nn.MSELoss()
        self.net = self.net.to(self.device)
        self.critic = self.critic.to(self.device)

    def store_transition(
        self,
        s,
        a,
        r,
        s_,
    ):  # 定义记忆存储函数 (这里输入为一个transition)

        self.memory_counter = self.memory_counter + 1
        if self.memory_counter < self.memory_capacity:
            self.s_memory.append(s)
            self.a_memory.append(a)
            self.r_memory.append(r)
            self.sn_memory.append(s_)
        else:
            number = self.memory_counter % self.memory_capacity
            self.s_memory[number - 1] = s
            self.a_memory[number - 1] = a
            self.r_memory[number - 1] = r
            self.sn_memory[number - 1] = s_

    def compute_single_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.net(state)
        action = action.detach().cpu().numpy()
        return action

    def learn(self):
        length = len(self.s_memory)
        out1 = random.sample(range(length), int(length / 10))
        # random sample
        s_learn = []
        a_learn = []
        r_learn = []
        sn_learn = []
        for number in out1:
            s_learn.append(self.s_memory[number])
            a_learn.append(self.a_memory[number])
            r_learn.append(self.r_memory[number])
            sn_learn.append(self.sn_memory[number])
        self.critic_learn_time = self.critic_learn_time + 1

        for bs, ba, br, bs_ in zip(s_learn, a_learn, r_learn, sn_learn):
            #update actor
            a = self.net(bs)
            q = self.critic(bs, a)
            a_loss = -torch.mean(q)
            self.optimizer_actor.zero_grad()
            a_loss.backward(retain_graph=True)
            self.optimizer_actor.step()
            #update critic
            a_ = self.net(bs_)
            q_ = self.critic(bs_, a_.detach())
            q_target = br + self.gamma * q_
            q_eval = self.critic(bs, ba.detach())
            # print(q_eval)
            # print(q_target)
            td_error = self.mse_loss(q_target.detach(), q_eval)
            # print(td_error)
            self.optimizer_critic.zero_grad()
            td_error.backward()
            self.optimizer_critic.step()

    def train_with_valid(self):
        rewards_list = []
        for i in range(self.num_epoch):
            j = 0
            done = False
            s = self.train_env_instance.reset()
            while not done:

                old_state = s
                action = self.net(torch.from_numpy(s).float())
                s, reward, done, _ = self.train_env_instance.step(
                    action.detach().numpy())
                self.store_transition(
                    torch.from_numpy(old_state).float().to(self.device),
                    action,
                    torch.tensor(reward).float().to(self.device),
                    torch.from_numpy(s).float().to(self.device))
                j = j + 1
                if j % 200 == 1:

                    self.learn()
            all_model_path = self.model_path + "/all_model/"
            best_model_path = self.model_path + "/best_model/"
            if not os.path.exists(all_model_path):
                os.makedirs(all_model_path)
            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)
            torch.save(self.net,
                       all_model_path + "actor_num_epoch_{}.pth".format(i))
            torch.save(self.critic,
                       all_model_path + "critic_num_epoch_{}.pth".format(i))
            s = self.valid_env_instance.reset()
            done = False
            rewards = 0
            while not done:

                old_state = s
                action = self.net(torch.from_numpy(s).float())
                s, reward, done, _ = self.valid_env_instance.step(
                    action.detach().numpy())
                rewards = rewards + reward
            rewards_list.append(rewards)
        index = rewards_list.index(np.max(rewards_list))
        actor_model_path = all_model_path + "actor_num_epoch_{}.pth".format(
            index)
        critic_model_path = all_model_path + "critic_num_epoch_{}.pth".format(
            index)
        self.net = torch.load(actor_model_path)
        self.critic = torch.load(critic_model_path)
        torch.save(self.net, best_model_path + "actor.pth")
        torch.save(self.critic, best_model_path + "critic.pth")

    def test(self):
        s = self.test_env_instance.reset()
        done = False
        while not done:
            old_state = s
            action = self.net(torch.from_numpy(s).float())
            s, reward, done, _ = self.test_env_instance.step(
                action.detach().numpy())
        df_return = self.test_env_instance.save_portfolio_return_memory()
        df_assets = self.test_env_instance.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        df.to_csv(self.result_path + "/result.csv")


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.autograd.set_detect_anomaly(True):
        a = trader(args)
        a.train_with_valid()
        a.test()
