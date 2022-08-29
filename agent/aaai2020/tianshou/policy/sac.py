import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import DDPGPolicy


class SACPolicy(DDPGPolicy):
    """docstring for SACPolicy"""

    def __init__(self, actor, actor_optim, critic1, critic1_optim,
                 critic2, critic2_optim, tau=0.005, gamma=0.99,
                 alpha=0.2, action_range=None, reward_normalization=False,
                 ignore_done=False):
        super().__init__(None, None, None, None, tau, gamma, 0,
                         action_range, reward_normalization, ignore_done)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._alpha = alpha
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self):
        self.training = True
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def sync_weight(self):
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def __call__(self, batch, state=None, input='obs'):
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = torch.distributions.Normal(*logits)

        x = dist.rsample()
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        log_prob = dist.log_prob(x) - torch.log(
            self._action_scale * (1 - y.pow(2)) + self.__eps)
        act = act.clamp(self._range[0], self._range[1])
        return Batch(
            logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch, batch_size=None, repeat=1):
        obs_next_result = self(batch, input='obs_next')
        a_ = obs_next_result.act
        dev = a_.device
        batch.act = torch.tensor(batch.act, dtype=torch.float, device=dev)
        target_q = torch.min(
            self.critic1_old(batch.obs_next, a_),
            self.critic2_old(batch.obs_next, a_),
        ) - self._alpha * obs_next_result.log_prob
        rew = torch.tensor(batch.rew, dtype=torch.float, device=dev)[:, None]
        done = torch.tensor(batch.done, dtype=torch.float, device=dev)[:, None]
        target_q = (rew + (1. - done) * self._gamma * target_q).detach()
        obs_result = self(batch)
        a = obs_result.act
        current_q1, current_q1a = self.critic1(
            np.concatenate([batch.obs, batch.obs]), torch.cat([batch.act, a])
        ).split(batch.obs.shape[0])
        current_q2, current_q2a = self.critic2(
            np.concatenate([batch.obs, batch.obs]), torch.cat([batch.act, a])
        ).split(batch.obs.shape[0])
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(
            current_q1a, current_q2a)).mean()
        # critic 1
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()
        # critic 2
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward(retain_graph=True)
        self.critic2_optim.step()
        # actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            'loss/actor': actor_loss.detach().cpu().numpy(),
            'loss/critic1': critic1_loss.detach().cpu().numpy(),
            'loss/critic2': critic2_loss.detach().cpu().numpy(),
        }
