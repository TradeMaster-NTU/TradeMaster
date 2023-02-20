import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr, GeneralReplayBuffer, get_optim_param
import torch
from torch import Tensor
from typing import Tuple
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from types import MethodType
from collections import namedtuple

@AGENTS.register_module()
class AlgorithmicTradingDQN(AgentBase):
    def __init__(self, **kwargs):
        super(AlgorithmicTradingDQN, self).__init__(**kwargs)

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.device = get_attr(kwargs, "device", torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_step = get_attr(kwargs,"max_step", 12345)  # the max step number of an episode. 'set as 12345 in default.

        self.state_dim = get_attr(kwargs, "state_dim", 10) # vector dimension (feature number) of state
        self.action_dim = get_attr(kwargs, "action_dim", 2) # vector dimension (feature number) of action

        '''Arguments for reward shaping'''
        self.gamma = get_attr(kwargs, "gamma", 0.99)  # discount factor of future rewards
        self.reward_scale = get_attr(kwargs, "reward_scale", 2 ** 0)  # an approximate target reward usually be closed to 256
        self.repeat_times = get_attr(kwargs, "repeat_times", 1.0)  # repeatedly update network using ReplayBuffer
        self.batch_size = int(get_attr(kwargs, "batch_size", 64))

        self.clip_grad_norm = get_attr(kwargs, "clip_grad_norm", 3.0)  # clip the gradient after normalization
        self.soft_update_tau = get_attr(kwargs, "soft_update_tau", 0)  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = get_attr(kwargs, "state_value_tau", 5e-3)  # the tau of normalize for value and state

        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)

        '''network'''
        self.act = self.act_target = get_attr(kwargs, "act", None).to(self.device)
        self.cri = self.cri_target = get_attr(kwargs, "cri", None) if get_attr(kwargs, "cri", None) else self.act

        '''optimizer'''
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None) if get_attr(kwargs, "cri_optimizer", None) else self.act_optimizer
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.criterion = get_attr(kwargs, "criterion", None)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.transition = get_attr(kwargs, "transition", namedtuple("Transition", ['state','action','reward','undone','next_state']))

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

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        next_states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(1, 1)) if if_random else get_action(state.unsqueeze(0))
            states[t] = state

            ary_action = action[0, 0].detach().cpu().numpy()
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

    def get_action(self, state: Tensor)-> Tensor:
        return self.act(state)

    def get_obj_critic(self, buffer: GeneralReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            transition = buffer.sample(batch_size)
            state = transition.state
            action = transition.action
            reward = transition.reward
            undone = transition.undone
            next_state = transition.next_state
            next_q = self.cri_target(next_state).max(dim=1, keepdim=True)[0].squeeze(1)
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state).gather(1, action.long()).squeeze(1)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def update_net(self, buffer: GeneralReplayBuffer) -> Tuple[float, ...]:
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))