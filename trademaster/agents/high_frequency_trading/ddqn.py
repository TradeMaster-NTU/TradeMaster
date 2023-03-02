import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr, get_optim_param, ReplayBufferHFT
import torch
from torch import Tensor
from typing import Tuple
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from types import MethodType
import torch.nn.functional as F


@AGENTS.register_module()
class HighFrequencyTradingDDQN(AgentBase):
    def __init__(self, **kwargs):
        super(HighFrequencyTradingDDQN, self).__init__(**kwargs)

        self.num_envs = int(
            get_attr(kwargs, "num_envs", 1)
        )  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.device = get_attr(
            kwargs,
            "device",
            torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        self.auxiliary_coffient = get_attr(
            kwargs,
            "auxiliary_coffient",
            512,
        )

        self.state_dim = get_attr(
            kwargs, "state_dim", 66
        )  # vector dimension (feature number) of state
        self.action_dim = get_attr(
            kwargs, "action_dim", 11
        )  # vector dimension (feature number) of action
        """Arguments for reward shaping"""
        self.gamma = get_attr(
            kwargs, "gamma", 0.99
        )  # discount factor of future rewards
        self.reward_scale = get_attr(
            kwargs, "reward_scale", 2**0
        )  # an approximate target reward usually be closed to 256
        self.repeat_times = get_attr(
            kwargs, "repeat_times", 1.0
        )  # repeatedly update network using ReplayBuffer
        self.batch_size = int(get_attr(kwargs, "batch_size", 512))
        self.target_update_freq = int(get_attr(kwargs, "target_update_freq", 10))

        self.clip_grad_norm = get_attr(
            kwargs, "clip_grad_norm", 3.0
        )  # clip the gradient after normalization
        self.soft_update_tau = get_attr(
            kwargs, "soft_update_tau", 0
        )  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = get_attr(
            kwargs, "state_value_tau", 5e-3
        )  # the tau of normalize for value and state

        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.last_info = None
        self.start_index = 0
        """network"""
        self.act = self.act_target = get_attr(kwargs, "act", None).to(self.device)
        self.cri = self.cri_target = (
            get_attr(kwargs, "cri", None) if get_attr(kwargs, "cri", None) else self.act
        )
        """optimizer"""
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = (
            get_attr(kwargs, "cri_optimizer", None)
            if get_attr(kwargs, "cri_optimizer", None)
            else self.act_optimizer
        )
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)
        """attribute"""
        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            raise NotImplementedError("Not support for multi env")

        self.criterion = get_attr(kwargs, "criterion", None)
        self.get_obj_critic = self.get_obj_critic_raw
        self.act_target = self.cri_target = deepcopy(self.act)
        self.completed_eposide_number = 0
        self.update_time = 0

    def get_action_test(self, ary_state, ary_info):
        state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
        previous_action = torch.as_tensor(
            ary_info["previous_action"], dtype=torch.long, device=self.device
        )
        avaliable_action = torch.as_tensor(
            ary_info["avaliable_action"], dtype=torch.float32, device=self.device
        )
        action = self.cri.get_action_test(
            state.unsqueeze(0),
            previous_action.unsqueeze(0),
            avaliable_action.unsqueeze(0),
        )
        action = action[0, 0].detach().cpu().numpy()
        return action

    def get_save(self):
        models = {
            "act": self.act,
            "cri": self.cri,
            "act_target": self.act_target,
            "cri_target": self.cri_target,
        }
        optimizers = {
            "act_optimizer": self.act_optimizer,
            "cri_optimizer": self.cri_optimizer,
        }
        res = {"models": models, "optimizers": optimizers}
        return res

    def explore_one_env(
        self,
        env,
        horizon_len,
        buffer: ReplayBufferHFT,
        start_list: list,
        if_random: bool = False,
    ) -> ReplayBufferHFT:
        ary_state = self.last_state
        # last_state.shape = (state_dim, ) for a single env.
        ary_info = self.last_info
        random_start = start_list[self.start_index]
        if ary_state is None:
            ary_state, ary_info = env.reset(random_start)
        get_action = self.cri.get_action
        for t in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            previous_action = torch.as_tensor(
                ary_info["previous_action"], dtype=torch.long, device=self.device
            )
            avaliable_action = torch.as_tensor(
                ary_info["avaliable_action"], dtype=torch.float32, device=self.device
            )
            action = (
                torch.randint(self.action_dim, size=(1, 1))
                if if_random
                else get_action(
                    state.unsqueeze(0),
                    previous_action.unsqueeze(0),
                    avaliable_action.unsqueeze(0),
                )
            )  # different

            ary_action = action[0, 0].detach().cpu().numpy()
            ary_state_, reward, done, ary_info_ = env.step(ary_action)  # next_state
            buffer.add(
                ary_state,
                ary_info,
                ary_action,
                reward,
                ary_state_,
                ary_info_,
                done,
            )
            ary_state, ary_info = ary_state_, ary_info_
            if done:
                ary_state, ary_info = env.reset(random_start)
                self.start_index += 1
                random_start = start_list[self.start_index]
                self.completed_eposide_number += 1
        self.last_info, self.last_state = ary_info_, ary_state_
        return buffer

    def get_obj_critic_raw(self, buffer: ReplayBufferHFT) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            (
                states,
                infos,
                actions,
                rewards,
                next_states,
                next_infos,
                dones,
            ) = buffer.sample()
            next_q = (
                self.cri_target(
                    next_states.reshape(buffer.batch_size, -1),
                    next_infos["previous_action"].long(),
                    next_infos["avaliable_action"],
                )
                .max(dim=1, keepdim=True)[0]
                .squeeze(1)
            )
            q_label = rewards + (1 - dones) * next_q

        q_value = (
            self.cri(
                states.reshape(buffer.batch_size, -1),
                infos["previous_action"].long(),
                infos["avaliable_action"],
            )
            .gather(1, actions.long())
            .squeeze(1)
        )
        q_distribution = self.cri(
            states.reshape(buffer.batch_size, -1),
            infos["previous_action"].long(),
            infos["avaliable_action"],
        )

        demonstration = infos["DP_action"]
        loss = self.criterion(q_value, q_label, q_distribution, demonstration)
        return loss

    def optimizer_update(self, optimizer: torch.optim, loss):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(
            parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm
        )
        optimizer.step()

    def update_net(self, buffer: ReplayBufferHFT) -> Tuple[float, ...]:
        update_times = int(1 * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            loss = self.get_obj_critic(buffer)

            self.optimizer_update(self.cri_optimizer, loss)
            self.update_time += 1
            if self.update_time % self.target_update_freq == 1:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return loss / update_times

    @staticmethod
    def soft_update(
        target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float
    ):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))
