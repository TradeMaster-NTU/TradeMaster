import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr, ReplayBuffer, get_optim_param
import numpy as np
import torch
from random import sample
from types import MethodType
from copy import deepcopy
from torch import Tensor
from typing import Tuple
from collections import namedtuple


@AGENTS.register_module()
class OrderExecutionETEO(AgentBase):
    def __init__(self, **kwargs):
        super(OrderExecutionETEO, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)

        '''network'''
        self.act = self.act_target = get_attr(kwargs, "act", None).to(self.device)
        self.cri = self.cri_target = get_attr(kwargs, "cri", None).to(self.device) if get_attr(kwargs, "cri", None) else self.act

        '''optimizer'''
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None) if get_attr(kwargs, "cri_optimizer", None) else self.act_optimizer
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)
        self.time_steps = get_attr(kwargs, "time_steps", 10)
        self.batch_size = int(get_attr(kwargs, "batch_size", 64))

        self.gamma = get_attr(kwargs, "gamma", 0.9)
        self.climp = get_attr(kwargs, "climp", 0.2)
        self.reward_scale = get_attr(kwargs, "reward_scale",
                                     2 ** 0)
        self.repeat_times = get_attr(kwargs, "repeat_times", 1.0)

        self.act_target = self.cri_target = deepcopy(self.act)

        self.transition = get_attr(kwargs, "transition",
                                   namedtuple("Transition", ['state', 'action', 'reward', 'undone', 'next_state']))

        self.last_state = None

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

    def init_states(self, env):
        states = np.zeros((self.time_steps, self.state_dim), dtype=np.float32)
        state = env.reset()
        states[0] = state
        for step in range(1, self.time_steps):
            action = np.array([0, 0])
            state, _, _, _ = env.step(action)
            states[step] = state
        return states

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        states = torch.zeros((horizon_len, self.num_envs, self.time_steps, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 2), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        next_states = torch.zeros((horizon_len, self.num_envs, self.time_steps, self.state_dim), dtype=torch.float32).to(self.device)

        state = self.last_state  # last_state.shape = (1, time_steps, state_dim) for a single env.
        get_action = self.get_action
        for t in range(horizon_len):
            action = get_action(state.unsqueeze(0))
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            states[t] = state

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            ary_state = torch.as_tensor(env.reset() if done else ary_state, dtype=torch.float32, device=self.device)

            # compress state
            state = torch.concat([state, ary_state.view(1, 1,-1)],dim=1)
            state = state[:, 1:,:]

            actions[t] = action
            rewards[t] = reward
            dones[t] = bool(done)
            next_states[t] = state

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)

        states = states.view((horizon_len, self.num_envs, self.time_steps, self.state_dim))

        transition = self.transition(
            state=states,
            action=actions,
            reward=rewards,
            undone=undones,
            next_state=next_states
        )
        return transition

    def get_action(self, state, if_train = True):

        action_volume, action_price, v = self.cri_target(state)
        action_volume = action_volume.squeeze()
        action_price = action_price.squeeze()

        if if_train:
            dis_volume = torch.distributions.normal.Normal(
                torch.relu(action_volume[0]) + 0.001,
                torch.relu(action_volume[1]) + 0.001)
            dis_price = torch.distributions.normal.Normal(
                torch.relu(action_price[0]) + 0.001,
                torch.relu(action_price[1]) + 0.001)
            volume = dis_volume.sample()
            price = dis_price.sample()
            action = np.array([torch.abs(volume).item(), torch.abs(price).item()])
        else:
            action = np.array([
                torch.relu(action_volume[0]).item() + 0.001,
                torch.relu(action_price[0]).item() + 0.001
            ])
        return action

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        update_times = int(self.repeat_times)
        assert update_times >= 1

        for _ in range(update_times):

            transition = buffer.sample(self.batch_size)
            state = transition.state
            action = transition.action
            reward = transition.reward
            undone = transition.undone
            next_state = transition.next_state

            action_volume, action_price, v = self.cri_target(next_state)
            td_target = reward + self.gamma * v * undone

            action_volume, action_price, v = self.cri_target(state)
            mean = torch.cat((action_volume[:, 0].unsqueeze(1),
                              action_price[:,0].unsqueeze(1)), dim=1)
            std = torch.cat((torch.relu(action_volume[:, 1].unsqueeze(1)) + 0.001,
                             torch.relu(action_price[:, 1].unsqueeze(1)) + 0.001), dim=1)
            old_dis = torch.distributions.normal.Normal(mean, std)

            log_prob_old = old_dis.log_prob(action).float()
            log_prob_old = (log_prob_old[:, 0] + log_prob_old[:, 1]).float()

            action_volume, action_price, v_s = self.act(next_state)
            action_volume, action_price, v = self.act(state)
            td_error = reward + self.gamma * v_s * undone - v

            action_volume, action_price, v = self.act(state)
            mean = torch.cat((action_volume[:, 0].unsqueeze(1),
                              action_price[:, 0].unsqueeze(1)), dim=1)
            std = torch.cat((torch.relu(action_volume[:, 1].unsqueeze(1)) + 0.001,
                             torch.relu(action_price[:, 1].unsqueeze(1)) + 0.001), dim=1)

            new_dis = torch.distributions.normal.Normal(mean, std)
            log_prob_new = new_dis.log_prob(action).float()
            log_prob_new = (log_prob_new[:, 0] + log_prob_new[:, 1]).float()

            ratio = torch.exp(torch.min(log_prob_new - log_prob_old, torch.tensor([10]).to(self.device)))

            L1 = ratio * td_error.float()
            L2 = torch.clamp(ratio, 1 - self.climp, 1 + self.climp) * td_error.float()
            loss_pi = -torch.min(L1, L2).mean().float()

            loss_v = torch.nn.functional.mse_loss(td_target.detach(), v.float())
            loss = loss_pi.float() + loss_v.float()

            self.act_optimizer.zero_grad()
            loss.backward()
            self.act_optimizer.step()

            self.cri_target.load_state_dict(self.act.state_dict(), strict=True)
