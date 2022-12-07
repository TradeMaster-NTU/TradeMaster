import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import os
import yaml
import argparse

parser = argparse.ArgumentParser()
sys.path.append(".")
from agent.oracle_distillation.model import Net
from env.OE.order_execution_for_PD import TradingEnv


class PPOtrainer:
    # PPO1 td error+KL td error is calculated using the new net times a factor calculated by both of the policy

    def __init__(self, input_fea, private_fea, hidden, lr, device) -> None:
        self.device = device
        self.net = Net(input_fea, hidden, private_fea).to(self.device)
        self.old_net = Net(input_fea, hidden, private_fea).to(self.device)
        self.old_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def choose_action(self, s_public, s_private):
        mu, sigma, V = self.old_net(s_public, s_private)
        dis = torch.distributions.normal.Normal(mu, sigma)  #构建分布
        a = dis.sample()  #采样出一个动作
        log_p = dis.log_prob(a)
        return a.item()

    def get_dis(self, s_public, s_private):
        mu, sigma, V = self.old_net(s_public, s_private)
        dis = torch.distributions.normal.Normal(mu, sigma)
        return dis

    def get_probablity_ratio(self, s_public, s_private, a):
        mu_old, sigma_old, _ = self.old_net(s_public, s_private)
        mu, sigma, _ = self.net(s_public, s_private)
        # print(mu_old.shape)
        # print(sigma_old.shape)
        new_dis = torch.distributions.normal.Normal(mu, sigma)
        old_dis = torch.distributions.normal.Normal(mu_old, sigma_old)
        new_prob = new_dis.log_prob(a).exp()
        # print(new_prob.shape)
        old_prob = old_dis.log_prob(a).exp()
        # print(old_prob.shape)
        return new_prob / (old_prob + 1e-12)

    def get_KL(self, s_public, s_private, a):
        mu_old, sigma_old, _ = self.old_net(s_public, s_private)
        mu, sigma, _ = self.net(s_public, s_private)
        new_dis = torch.distributions.normal.Normal(mu, sigma)
        old_dis = torch.distributions.normal.Normal(mu_old, sigma_old)
        kl = torch.distributions.kl.kl_divergence(new_dis, old_dis)
        return kl

    def choose_action_test(self, s_public, s_private):
        with torch.no_grad():
            mu, sigma, V = self.old_net(s_public, s_private)
        return mu.cpu().squeeze().detach().numpy()

    def uniform(self):
        self.old_net.load_state_dict(self.net.state_dict())


if __name__ == "__main__":
    # kl = torch.distributions.kl.kl_divergence(
    #     torch.distributions.normal.Normal(1, 20),
    #     torch.distributions.normal.Normal(1, 2))
    # print(kl)
    input = torch.randn(1, 10, 11).to("cuda")
    private_state = torch.randn(1, 10, 2).to("cuda")
    net = PPOtrainer(11, 2, 16, 1e-3, "cuda")
    a, log_p, dis = net.choose_action(input, private_state)
    print(a)
