import os
import random
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr, get_optim_param
import numpy as np
import torch
from types import MethodType
from copy import deepcopy
from collections import namedtuple
from torch import Tensor
from typing import Tuple

class PPOtrainer:
    # PPO1 td error+KL td error is calculated using the new net times a factor calculated by both of the policy

    def __init__(self, net, old_net, optimizer):
        self.net = net
        self.old_net = old_net
        self.old_net.load_state_dict(self.net.state_dict())
        self.optimizer = optimizer

    def choose_action(self, s_public, s_private):
        if len(s_public.shape) >= 4:
            s_public = s_public.squeeze(1)
        if len(s_private.shape) >= 4:
            s_private = s_private.squeeze(1)

        mu, sigma, V = self.old_net(s_public, s_private)
        dis = torch.distributions.normal.Normal(mu, sigma)
        a = dis.sample()
        log_p = dis.log_prob(a)
        return a

    def get_dis(self, s_public, s_private):
        mu, sigma, V = self.old_net(s_public, s_private)
        dis = torch.distributions.normal.Normal(mu, sigma)
        return dis

    def get_probablity_ratio(self, s_public, s_private, a):
        mu_old, sigma_old, _ = self.old_net(s_public, s_private)
        mu, sigma, _ = self.net(s_public, s_private)
        new_dis = torch.distributions.normal.Normal(mu, sigma)
        old_dis = torch.distributions.normal.Normal(mu_old, sigma_old)
        new_prob = new_dis.log_prob(a).exp()

        old_prob = old_dis.log_prob(a).exp()
        res  = new_prob / (old_prob + 1e-12)
        return res.view(-1)

    def get_KL(self, s_public, s_private, a):
        mu_old, sigma_old, _ = self.old_net(s_public, s_private)
        mu, sigma, _ = self.net(s_public, s_private)
        new_dis = torch.distributions.normal.Normal(mu, sigma)
        old_dis = torch.distributions.normal.Normal(mu_old, sigma_old)
        kl = torch.distributions.kl.kl_divergence(new_dis, old_dis)
        return kl.view(-1)

    def choose_action_test(self, s_public, s_private):
        with torch.no_grad():
            mu, sigma, V = self.old_net(s_public, s_private)
        return mu.cpu().squeeze().detach().numpy()

    def uniform(self):
        self.old_net.load_state_dict(self.net.state_dict())


@AGENTS.register_module()
class OrderExecutionPD(AgentBase):
    def __init__(self, **kwargs):
        super(OrderExecutionPD, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)

        '''network'''
        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None).to(self.device) if get_attr(kwargs, "cri", None) else self.act
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        '''optimizer'''
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None) if get_attr(kwargs, "cri_optimizer",
                                                                                 None) else self.act_optimizer
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.student_ppo = PPOtrainer(net=self.act, old_net=self.act_target, optimizer=self.act_optimizer)
        self.teacher_ppo = PPOtrainer(net=self.cri, old_net=self.cri_target, optimizer=self.cri_optimizer)

        self.criterion = get_attr(kwargs, "criterion", None)

        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)
        self.public_state_dim = get_attr(kwargs, "public_state_dim", None)
        self.private_state_dim = get_attr(kwargs, "private_state_dim", None)
        self.time_steps = get_attr(kwargs, "time_steps", 10)
        self.batch_size = int(get_attr(kwargs, "batch_size", 64))

        self.gamma = get_attr(kwargs, "gamma", 0.9)
        self.beta = get_attr(kwargs, "beta", 1)
        self.lambada = get_attr(kwargs, "lambada", 1)
        self.reward_scale = get_attr(kwargs, "reward_scale",2 ** 0)
        self.repeat_times = get_attr(kwargs, "repeat_times", 1.0)

        self.transition = get_attr(kwargs, "transition",
                                   namedtuple("TransitionPD", ['state',
                                                               'action',
                                                               'reward',
                                                               'undone',
                                                               'next_state',
                                                               'public_state',
                                                               'private_state',
                                                               'next_public_state',
                                                               'next_private_state'
                                                               ]))

        self.last_state = None
        self.last_public_state = None
        self.last_private_state = None

    def get_save(self):
        models = {
            "act":self.act,
            "cri":self.cri,
            "act_target": self.act_target,
            "cri_target": self.cri_target,
        }
        optimizers = {
            "act_optimizer":self.act_optimizer,
            "cri_optimizer":self.cri_optimizer
        }
        res = {
            "models":models,
            "optimizers":optimizers
        }
        return res

    def get_teacher_action(self, public_state, private_state):
        return self.teacher_ppo.choose_action(public_state, private_state)

    def get_student_action(self, public_state, private_state):
        return self.student_ppo.choose_action(public_state, private_state)

    def explore_env(self, env, horizon_len: int, if_teacher: bool) -> Tuple[Tensor, ...]:
        states = torch.zeros((horizon_len, self.num_envs, self.time_steps, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        next_states = torch.zeros((horizon_len, self.num_envs, self.time_steps, self.state_dim), dtype=torch.float32).to(
            self.device)
        public_states = torch.zeros((horizon_len, self.num_envs, self.time_steps * 2, self.public_state_dim),
                                  dtype=torch.float32).to(self.device)
        private_states = torch.zeros((horizon_len, self.num_envs, self.time_steps, self.private_state_dim),
                                  dtype=torch.float32).to(self.device)
        next_public_states = torch.zeros((horizon_len, self.num_envs, self.time_steps * 2, self.public_state_dim),
                                  dtype=torch.float32).to(self.device)
        next_private_states = torch.zeros((horizon_len, self.num_envs, self.time_steps, self.private_state_dim),
                                  dtype=torch.float32).to(self.device)

        state = self.last_state
        public_state = self.last_public_state
        private_state = self.last_private_state

        if if_teacher:
            get_action = self.get_teacher_action
        else:
            get_action = self.get_student_action

        for t in range(horizon_len):

            if if_teacher:
                action = get_action(public_state.unsqueeze(0), private_state.unsqueeze(0))
            else:
                action = get_action(state.unsqueeze(0), private_state.unsqueeze(0))
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

            states[t] = state
            public_states[t] = public_state
            private_states[t] = private_state

            ary_action = action[0, 0].detach().cpu().numpy()
            ary_state, reward, done, ary_info = env.step(ary_action)
            if done:
                ary_state, ary_info = env.reset()

            ary_state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            ary_public_state = torch.as_tensor(ary_info["perfect_state"], dtype=torch.float32, device=self.device)
            ary_private_state = torch.as_tensor(ary_info["private_state"], dtype=torch.float32, device=self.device)

            state = ary_state
            public_state = ary_public_state
            private_state = ary_private_state

            actions[t] = action
            rewards[t] = reward
            dones[t] = bool(done)
            next_states[t] = state
            next_public_states[t] = public_state
            next_private_states[t] = private_state

        self.last_state = state
        self.last_public_state = public_state
        self.last_private_state =private_state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)

        transition = self.transition(
            state=states,
            action=actions,
            reward=rewards,
            undone=undones,
            next_state=next_states,
            public_state = public_states,
            private_state = private_states,
            next_public_state = next_public_states,
            next_private_state = next_private_states,
        )
        return transition

    def update_teacher(self, buffer):

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1

        for _ in range(update_times):
            transition = buffer.sample(self.batch_size)
            public_state = transition.public_state
            private_state = transition.private_state
            action = transition.action
            reward = transition.reward
            next_public_state = transition.next_public_state
            next_private_state = transition.next_private_state
            undone = transition.undone

            advangetage = reward + self.gamma * self.teacher_ppo.net.get_V(next_public_state, next_private_state) * undone
            advangetage = advangetage - self.teacher_ppo.net.get_V(public_state, private_state)

            log_ratio = self.teacher_ppo.get_probablity_ratio(next_public_state, next_private_state, action)
            kl = self.teacher_ppo.get_KL(next_public_state, next_private_state, action)

            loss = -(advangetage * log_ratio - self.beta * kl).mean()
            self.teacher_ppo.optimizer.zero_grad()
            loss.backward()
            self.teacher_ppo.optimizer.step()

            self.teacher_ppo.uniform()

    def update_student(self, buffer):
        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1

        for _ in range(update_times):
            transition = buffer.sample(self.batch_size)
            state = transition.state
            public_state = transition.public_state
            private_state = transition.private_state
            action = transition.action
            reward = transition.reward
            next_state = transition.next_state
            next_public_state = transition.next_public_state
            next_private_state = transition.next_private_state
            undone = transition.undone

            advangetage = reward + self.gamma * self.student_ppo.net.get_V(next_state,
                                                                           next_private_state) * undone
            advangetage = advangetage - self.student_ppo.net.get_V(state, private_state)

            log_ratio = self.student_ppo.get_probablity_ratio(next_state, next_private_state, action)
            kl = self.student_ppo.get_KL(next_state, next_private_state, action)

            teacher_dis = self.teacher_ppo.get_dis(public_state,
                                                   next_private_state)
            student_dis = self.student_ppo.get_dis(next_state,
                                                   next_private_state)

            loss = -(advangetage * log_ratio - self.beta * kl - self.lambada * torch.distributions.kl.kl_divergence(teacher_dis, student_dis)).mean()
            self.student_ppo.optimizer.zero_grad()
            loss.backward()
            self.student_ppo.optimizer.step()

            self.student_ppo.uniform()

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
