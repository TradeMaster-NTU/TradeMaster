import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

import torch
from torch import Tensor
from typing import Tuple
from ..builder import AGENTS
from ..custom import AgentBase
import random
from collections import namedtuple
from trademaster.utils import get_attr, GeneralReplayBuffer, get_optim_param



@AGENTS.register_module()
class PortfolioManagementEIIE(AgentBase):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIE, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_step = get_attr(kwargs, "max_step",
                                 12345)  # the max step number of an episode. 'set as 12345 in default.
        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)
        self.time_steps = get_attr(kwargs, "time_steps", 10)

        '''Arguments for reward shaping'''
        self.gamma = get_attr(kwargs, "gamma", 0.99)  # discount factor of future rewards
        self.reward_scale = get_attr(kwargs, "reward_scale",
                                     2 ** 0)  # an approximate target reward usually be closed to 256
        self.repeat_times = get_attr(kwargs, "repeat_times", 1.0)  # repeatedly update network using ReplayBuffer
        self.batch_size = int(get_attr(kwargs, "batch_size", 64))
        self.clip_grad_norm = get_attr(kwargs, "clip_grad_norm", 3.0)  # clip the gradient after normalization
        self.soft_update_tau = get_attr(kwargs, "soft_update_tau",
                                        0)  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = get_attr(kwargs, "state_value_tau", 5e-3)  # the tau of normalize for value and state

        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)

        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None).to(self.device)
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)

        self.criterion = get_attr(kwargs, "criterion", None)

        self.transition = get_attr(kwargs, "transition", namedtuple("Transition", ['state','action','reward','undone','next_state']))

    def get_save(self):
        models = {
            "act":self.act,
            "cri":self.cri
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

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        states = torch.zeros((horizon_len,
                              self.num_envs,
                              self.action_dim,
                              self.time_steps,
                              self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim + 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        next_states = torch.zeros((horizon_len,
                                   self.num_envs,
                                   self.action_dim,
                                   self.time_steps,
                                   self.state_dim), dtype=torch.float32).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        get_action = self.act
        for t in range(horizon_len):
            action = get_action(state.unsqueeze(0))
            states[t] = state

            ary_action = action[0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            state = torch.as_tensor(env.reset() if done else ary_state, dtype=torch.float32, device=self.device)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)

        transition = self.transition(
            state = states,
            action = actions,
            reward = rewards,
            undone = undones,
            next_state = next_states
        )
        return transition

    def update_net(self, buffer: GeneralReplayBuffer):
        obj_critics = 0.0
        obj_actors = 0.0
        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer: GeneralReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        transition = buffer.sample(self.batch_size)
        state = transition.state
        action = transition.action
        reward = transition.reward
        undone = transition.undone
        next_state = transition.next_state

        a = self.act(state)
        q = self.cri(state, a)
        a_loss = -torch.mean(q)

        self.act_optimizer.zero_grad()
        a_loss.backward()
        self.act_optimizer.step()

        a_ = self.act(next_state)
        q_ = self.cri(next_state, a_.detach())
        q_target = reward + self.gamma * q_
        q_eval = self.cri(state, action.detach())

        td_error = self.criterion(q_target.detach(), q_eval)

        self.cri_optimizer.zero_grad()
        td_error.backward()
        self.cri_optimizer.step()

        return td_error, q_target