from logging import raiseExceptions
from stat import S_ENFMT
import torch.nn as nn
import pandas as pd
import sys
import argparse
import random

sys.path.append(".")
from env.OE.order_execution_for_PD import TradingEnv
from agent.oracle_distillation.model import Net
from agent.oracle_distillation.ppo import PPOtrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
from agent.oracle_distillation.ppo import PPOtrainer

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed",
                    type=int,
                    default=12345,
                    help="the path for storing the downloaded data")
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="the value of gamma during training")
parser.add_argument("--train_env_config",
                    type=str,
                    default="config/input_config/env/OE/OE_for_PD/train.yml",
                    help="the path for train env config")
parser.add_argument("--valid_env_config",
                    type=str,
                    default="config/input_config/env/OE/OE_for_PD/valid.yml",
                    help="the path for valid env config")
parser.add_argument("--test_env_config",
                    type=str,
                    default="config/input_config/env/OE/OE_for_PD/test.yml",
                    help="the path for test env config")
parser.add_argument("--model_path",
                    type=str,
                    default="result/OPD/trained_model/",
                    help="the path for storing model")
parser.add_argument("--result_path",
                    type=str,
                    default="result/OPD/result/",
                    help="the path for storing result")
parser.add_argument("--memory_capacity",
                    type=int,
                    default=100,
                    help="the number of transcation you can store at one time")
parser.add_argument(
    "--memory_update_freq",
    type=int,
    default=10,
    help="the number of update before we clear all of the memory")
parser.add_argument("--hidden_nodes",
                    type=int,
                    default=32,
                    help="the number of hidden nodes in ")
parser.add_argument("--lr",
                    type=float,
                    default=1e-3,
                    help="the learning rate for the student and teacher")
parser.add_argument("--num_epoch",
                    type=int,
                    default=1000,
                    help="the number of epoch")
parser.add_argument("--beta",
                    type=float,
                    default=1,
                    help="the value of beta for ppo training")
parser.add_argument("--lambada",
                    type=float,
                    default=1,
                    help="the value of lambda for ppo student training")
parser.add_argument("--update_freq",
                    type=int,
                    default=1000,
                    help="the value of update_freq for saving a student ppo")


def load_yaml(dict):
    realpath = os.path.abspath(".")
    file_dict = os.path.join(realpath, dict)
    with open(file_dict, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class trader:
    def __init__(self, args) -> None:
        self.memory_student = []
        self.memory_teacher = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.memory_capacity = args.memory_capacity
        self.memory_update_freq = args.memory_update_freq
        self.train_env = TradingEnv(load_yaml(args.train_env_config))
        self.valid_env = TradingEnv(load_yaml(args.valid_env_config))
        self.test_env = TradingEnv(load_yaml(args.test_env_config))
        self.num_epoch = args.num_epoch
        self.set_seed(args.random_seed)
        self.input_fea = self.train_env.observation_space.shape[-1]
        _, info = self.train_env.reset()
        self.private_fea = info["private_state"].shape[-1]
        self.teacher_ppo = PPOtrainer(self.input_fea, self.private_fea,
                                      args.hidden_nodes, args.lr, self.device)
        self.student_ppo = PPOtrainer(self.input_fea, self.private_fea,
                                      args.hidden_nodes, args.lr, self.device)
        self.gamma = args.gamma
        self.beta = args.beta
        self.lambada = args.lambada
        self.save_freq = args.update_freq
        self.model_path = args.model_path
        self.result_path = args.result_path
        self.all_model_path = os.path.join(args.model_path, "all_model")
        self.best_model_path = os.path.join(args.model_path, "best_model")
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.all_model_path):
            os.makedirs(self.all_model_path)
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)

    def train_with_valid(self):
        valid_score_list = []
        self.step_teacher = 0
        self.step_student = 0
        valid_number = 0
        for i in range(self.num_epoch):
            s, info = self.train_env.reset()
            # train the teacher first
            done = False
            while not done:
                public_state = torch.from_numpy(info["perfect_state"]).to(
                    self.device).float()
                private_state = torch.from_numpy(info["private_state"]).to(
                    self.device).float()
                self.step_teacher += 1

                action = self.teacher_ppo.choose_action(
                    public_state, private_state)
                s, r, done, info_ = self.train_env.step(action)
                self.store_transcation_teacher(info, action, r, info_, done)
                info = info_
                if self.step_teacher % self.memory_capacity == 1:
                    print("teacher learning")
                    self.teacher_learn()
            #then train the student
            s, info = self.train_env.reset()
            done = False
            while not done:
                public_state = torch.from_numpy(s).to(self.device).float()
                private_state = torch.from_numpy(info["private_state"]).to(
                    self.device).float()
                self.step_student += 1

                action = self.student_ppo.choose_action(
                    public_state, private_state)
                s_, r, done, info_ = self.train_env.step(action)
                self.store_transcation_student(s, info, action, r, s_, info_,
                                               done)
                info = info_
                s = s_

                if self.step_student % self.memory_capacity == 1:
                    print("student learning")
                    self.student_learn()

                if self.step_student % self.save_freq == 1:
                    torch.save(
                        self.student_ppo.old_net, self.all_model_path +
                        "/{}_net.pth".format(valid_number))
                    valid_number += 1
                    s, info = self.valid_env.reset()
                    done = False
                    while not done:
                        public_state = torch.from_numpy(s).to(
                            self.device).float()

                        private_state = torch.from_numpy(
                            info["private_state"]).to(self.device).float()
                        action = self.student_ppo.choose_action_test(
                            public_state, private_state)

                        s_, r, done, info_ = self.valid_env.step(action)
                        info = info_
                        s = s_
                    valid_score_list.append(self.valid_env.money_sold)
                    break
        index = valid_score_list.index(max(valid_score_list))
        net_path = self.all_model_path + "/{}_net.pth".format(index)
        self.student_ppo.old_net = torch.load(net_path)
        torch.save(self.student_ppo.old_net,
                   self.best_model_path + "/best_net.pth")

    def test(self):
        s, info = self.test_env.reset()
        action_list = []
        reward_list = []

        done = False
        while not done:
            public_state = torch.from_numpy(s).to(self.device).float()
            private_state = torch.from_numpy(info["private_state"]).to(
                self.device).float()
            action = self.student_ppo.choose_action_test(
                public_state, private_state)
            s_, r, done, info_ = self.test_env.step(action)
            info = info_
            s = s_
            action_list.append(action)
            reward_list.append(r)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        np.save(self.result_path + "/action.npy", action_list)
        np.save(self.result_path + "/reward.npy", reward_list)

    def store_transcation_teacher(self, info, a, r, info_, done):
        self.memory_teacher.append((
            torch.from_numpy(info["perfect_state"]).to(self.device).float(),
            torch.from_numpy(info["private_state"]).to(self.device).float(),
            torch.tensor([a]).to(self.device).float(),
            torch.tensor([r]).to(self.device).float(),
            torch.from_numpy(info_["perfect_state"]).to(self.device).float(),
            torch.from_numpy(info_["private_state"]).to(self.device).float(),
            torch.tensor([done]).to(self.device).float(),
        ))

    def teacher_learn(self):
        perfect_state_list = []
        private_state_list = []
        a_list = []
        r_list = []
        perfect_n_state_list = []
        private_n_state_list = []
        done_list = []
        for perfect_state, private_state, a, r, perfect_n_state, private_n_state, done in self.memory_teacher:
            advangetage = (
                r + ((self.gamma * self.teacher_ppo.net.get_V(
                    perfect_n_state, private_n_state)).squeeze() *
                     (1 - done).squeeze()) - (self.teacher_ppo.net.get_V(
                         perfect_state, private_state)).squeeze()).squeeze()
            log_ratio = self.teacher_ppo.get_probablity_ratio(
                perfect_n_state, private_n_state, a)
            # print(log_ratio)
            kl = self.teacher_ppo.get_KL(perfect_n_state, private_n_state, a)
            loss = -(advangetage * log_ratio - self.beta * kl)
            self.teacher_ppo.optimizer.zero_grad()
            loss.backward()
            self.teacher_ppo.optimizer.step()
        self.teacher_ppo.uniform()
        if self.step_teacher % self.memory_update_freq == 1:
            self.memory_teacher = []

        # print(log_ratio)
        #     perfect_state_list.append(perfect_state)
        #     private_state_list.append(private_state)
        #     a_list.append(a)
        #     r_list.append(r)
        #     perfect_n_state_list.append(perfect_n_state)
        #     private_n_state_list.append(private_n_state)
        #     done_list.append(done)
        # perfect_state = torch.cat(perfect_state_list, dim=0)
        # private_state = torch.cat(private_state_list, dim=0)
        # a = torch.cat(a_list, dim=0)
        # r = torch.cat(r_list, dim=0)
        # perfect_n_state = torch.cat(perfect_n_state_list, dim=0)
        # private_n_state = torch.cat(private_n_state_list, dim=0)
        # done = torch.cat(done_list, dim=0)

        # print((self.gamma *
        #        self.teacher_ppo.net.get_V(perfect_n_state, private_n_state) *
        #        (1 - done)).squeeze().shape)
        # print((self.gamma * self.teacher_ppo.net.get_V(
        #     perfect_n_state, private_n_state)).squeeze().shape)
        # print(((self.gamma * self.teacher_ppo.net.get_V(
        #     perfect_n_state, private_n_state)).squeeze() *
        #        (1 - done).squeeze()).shape)

        # advangetage = r + ((self.gamma * self.teacher_ppo.net.get_V(
        #     perfect_n_state, private_n_state)).squeeze() *
        #                    (1 - done).squeeze()) - (self.teacher_ppo.net.get_V(
        #                        perfect_state, private_state)).squeeze()
        # log_ratio = self.teacher_ppo.get_probablity_ratio(
        #     perfect_n_state, private_n_state, a)
        # print(log_ratio.shape)
    def student_learn(self):
        perfect_state_list = []
        private_state_list = []
        a_list = []
        r_list = []
        perfect_n_state_list = []
        private_n_state_list = []
        done_list = []
        for imperfect_state, private_state, perfect_state, a, r, imperfect_n_state, private_n_state, perfect_n_state, done in self.memory_student:
            advangetage = (
                r + ((self.gamma * self.student_ppo.net.get_V(
                    imperfect_n_state, private_n_state)).squeeze() *
                     (1 - done).squeeze()) - (self.student_ppo.net.get_V(
                         imperfect_state, private_state)).squeeze()).squeeze()
            log_ratio = self.student_ppo.get_probablity_ratio(
                imperfect_n_state, private_n_state, a)
            # print(log_ratio)
            kl = self.student_ppo.get_KL(imperfect_n_state, private_n_state, a)
            teacher_dis = self.teacher_ppo.get_dis(perfect_state,
                                                   private_n_state)
            student_dis = self.student_ppo.get_dis(imperfect_n_state,
                                                   private_n_state)
            loss = -(
                advangetage * log_ratio - self.beta * kl - self.lambada *
                torch.distributions.kl.kl_divergence(teacher_dis, student_dis))
            self.student_ppo.optimizer.zero_grad()
            loss.backward()
            self.student_ppo.optimizer.step()
        self.student_ppo.uniform()
        if self.step_student % self.memory_update_freq == 1:
            self.memory_student = []

    def store_transcation_student(self, s, info, a, r, s_, info_, done):
        self.memory_student.append(
            (torch.from_numpy(s).to(self.device).float(),
             torch.from_numpy(info["private_state"]).to(self.device).float(),
             torch.from_numpy(info["perfect_state"]).to(self.device).float(),
             torch.tensor([a]).to(self.device).float(),
             torch.tensor([r]).to(self.device).float(),
             torch.from_numpy(s_).to(self.device).float(),
             torch.from_numpy(info_["private_state"]).to(self.device).float(),
             torch.from_numpy(info_["perfect_state"]).to(self.device).float(),
             torch.tensor([done]).to(self.device).float()))

    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # args = parser.parse_args()
    # agent = trader(args)
    # agent.train_with_valid()
    # agent.test()
    action = np.load("result/OPD/result/reward.npy")
    print(action)