import sys
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr
import torch
from torch.distributions import Normal
import random
import pandas as pd
import numpy as np
from collections import namedtuple
from torch import Tensor
from typing import Tuple

def make_market_information(df, technical_indicator):
    #based on the information, calculate the average for technical_indicator to present the market average
    all_dataframe_list = []
    index_list = df.index.unique().tolist()
    index_list.sort()
    for i in index_list:
        information = df[df.index == i]
        new_dataframe = []
        for tech in technical_indicator:
            tech_value = np.mean(information[tech])
            new_dataframe.append(tech_value)
        all_dataframe_list.append(new_dataframe)
    new_df = pd.DataFrame(all_dataframe_list,
                          columns=technical_indicator).values
    # new_df.to_csv(store_path)
    return new_df


def make_correlation_information(df: pd.DataFrame, feature="adjclose"):
    # based on the information, we are making the correlation matrix(which is N*N matric where N is the number of tickers) based on the specific
    # feature here,  as default is adjclose
    df.sort_values(by='tic', ascending=True, inplace=True)
    array_symbols = df['tic'].values

    # get data, put into dictionary then dataframe
    dict_sym_ac = {}  # key=symbol, value=array of adj close
    for sym in array_symbols:
        dftemp = df[df['tic'] == sym]
        dict_sym_ac[sym] = dftemp['adjcp'].values

    # create correlation coeff df
    dfdata = pd.DataFrame.from_dict(dict_sym_ac)
    dfcc = dfdata.corr().round(2)
    dfcc = dfcc.values
    return dfcc

def generate_portfolio(scores=torch.sigmoid(torch.randn(29, 1)), quantile=0.5):
    scores = scores.squeeze()
    length = len(scores)
    if scores.equal(torch.ones(length)):
        weights = (1 / length) * torch.ones(length)
        return weights
    if scores.equal(torch.zeros(length)):
        weights = (-1 / length) * torch.ones(length)
        return weights
    sorted_score, indices = torch.sort(scores, descending=True)
    length = len(scores)
    rank_hold = int(quantile * length)
    value_hold = sorted_score[-1] + (sorted_score[0] -
                                     sorted_score[-1]) * quantile

    good_portfolio = []
    good_scores = []
    bad_portfolio = []
    bad_scores = []
    for i in range(length):
        score = scores[i]
        if score <= value_hold:
            bad_portfolio.append(i)
            bad_scores.append(score.unsqueeze(0))
        else:
            good_portfolio.append(i)
            good_scores.append(score.unsqueeze(0))
    final_portfollio = [0] * length
    good_scores = torch.cat(good_scores)
    bad_scores = torch.cat(bad_scores)
    good_portion = torch.exp(good_scores) / torch.sum(
        torch.exp(good_scores)) * (quantile)
    bad_portion = -torch.exp(1 - bad_scores) / torch.sum(
        torch.exp(1 - bad_scores)) * (1 - quantile)
    for i in range(length):
        if i in bad_portfolio:
            index = bad_portfolio.index(i)
            final_portfollio[i] = bad_portion[index]
        else:
            index = good_portfolio.index(i)
            final_portfollio[i] = good_portion[index]
    weights = []
    for weight in final_portfollio:
        weight_tensor = torch.tensor([weight])
        weights.append(weight_tensor)
    weights = torch.cat(weights)

    return weights

def generate_rho(mean: torch.tensor, std: torch.tensor):
    normal = Normal(mean, std)
    result = normal.sample()
    if result <= 0:
        result = torch.tensor(0)
    if result >= 1:
        result = torch.tensor(0.99)
    return result

@AGENTS.register_module()
class PortfolioManagementDeepTrader(AgentBase):
    def __init__(self, **kwargs):
        super(PortfolioManagementDeepTrader, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)

        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None).to(self.device)
        self.market = get_attr(kwargs, "market", None).to(self.device)

        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)
        self.market_optimizer = get_attr(kwargs, "market_optimizer", None)
        self.time_steps = get_attr(kwargs, "time_steps", 10)

        self.criterion = get_attr(kwargs, "criterion", None)
        self.transition = get_attr(kwargs, "transition",
                                   namedtuple("TransitionDeepTrader",
                                              ['state',
                                               'action',
                                               'reward',
                                               'undone',
                                               'next_state',
                                               'correlation_matrix',
                                               'next_correlation_matrix',
                                               'state_market',
                                               'roh_bar_market'
                                               ]))

        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)

        self.memory_counter = 0  # for storing memory
        self.memory_capacity = get_attr(kwargs, "memory_capacity", 1000)
        self.gamma = get_attr(kwargs, "gamma", 0.9)

        self.policy_update_frequency = get_attr(kwargs, "policy_update_frequency", 500)
        self.critic_learn_time = 0

    def get_save(self):
        models = {
            "act":self.act,
            "cri":self.cri,
            "market":self.market
        }
        optimizers = {
            "act_optimizer":self.act_optimizer,
            "cri_optimizer":self.cri_optimizer,
            "market_optimizer":self.market_optimizer
        }
        res = {
            "models":models,
            "optimizers":optimizers
        }
        return res

    def get_action(self, state, state_market, corr_matrix):
        asset_scores = self.act(state, corr_matrix)
        output_market = self.market(state_market)
        roh_bar = generate_rho(output_market[0].cpu(), output_market[1].cpu())
        weights = generate_portfolio(asset_scores.cpu(), roh_bar)
        action = weights.numpy()
        return action

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
        correlation_matrixs = torch.zeros((
            horizon_len, self.num_envs, self.action_dim, self.action_dim
        ), dtype=torch.float32).to(self.device)
        next_correlation_matrixs = torch.zeros((
            horizon_len, self.num_envs, self.action_dim, self.action_dim
        ), dtype=torch.float32).to(self.device)
        state_markets = torch.zeros((horizon_len,
                              self.num_envs,
                              self.time_steps,
                              self.state_dim), dtype=torch.float32).to(self.device)
        roh_bar_markets = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.

        get_action = self.get_action

        for t in range(horizon_len):
            market_state = torch.from_numpy(make_market_information(env.data,
                           technical_indicator=env.tech_indicator_list)).unsqueeze(
                0).float().to(self.device)
            corr_matrix = make_correlation_information(env.data)
            action = get_action(state,market_state,corr_matrix)
            print(action.shape)
            exit()


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

    def store_transition(self, s_asset,
                         a_asset,
                         r,
                         sn_asset,
                         s_market,
                         a_market,
                         sn_market,
                         A,
                         A_n,
                         roh_bar):  # 定义记忆存储函数 (这里输入为两套transition：asset和market)

        self.memory_counter = self.memory_counter + 1
        if self.memory_counter < self.memory_capacity:
            self.s_memory_asset.append(s_asset)
            self.a_memory_asset.append(a_asset)
            self.r_memory_asset.append(r)
            self.sn_memory_asset.append(sn_asset)
            self.correlation_matrix.append(A)
            self.correlation_n_matrix.append(A_n)

            self.s_memory_market.append(s_market)
            self.a_memory_market.append(a_market)
            self.r_memory_market.append(r)
            self.sn_memory_market.append(sn_market)
            self.roh_bars.append(roh_bar)

        else:
            number = self.memory_counter % self.memory_capacity
            self.s_memory_asset[number - 1] = s_asset
            self.a_memory_asset[number - 1] = a_asset
            self.r_memory_asset[number - 1] = r
            self.sn_memory_asset[number - 1] = sn_asset
            self.correlation_matrix[number - 1] = A
            self.correlation_n_matrix[number - 1] = A_n

            self.s_memory_market[number - 1] = s_market
            self.a_memory_market[number - 1] = a_market
            self.r_memory_market[number - 1] = r
            self.sn_memory_market[number - 1] = sn_market
            self.roh_bars[number - 1] = roh_bar

    def compute_weights_test(self, asset_state, market_state, A):
        # use the mean to compute roh
        asset_state = torch.from_numpy(asset_state).float().to(self.device)
        asset_scores = self.act(asset_state, A)
        input_market = torch.from_numpy(market_state).unsqueeze(0).to(
            torch.float32).to(self.device)
        output_market = self.market(input_market)
        weights = generate_portfolio(asset_scores.cpu().detach(),
                                     output_market[0].cpu().detach().numpy())
        weights = weights.detach().numpy()
        return weights

    def learn(self):
        length = len(self.s_memory_asset)
        out1 = random.sample(range(length), int(length / 10))
        # random sample
        s_learn_asset = []
        a_learn_asset = []
        r_learn_asset = []
        sn_learn_asset = []
        correlation_asset = []
        correlation_asset_n = []

        s_learn_market = []
        a_learn_market = []
        r_learn_market = []
        sn_learn_market = []
        roh_bar_market = []
        for number in out1:
            s_learn_asset.append(self.s_memory_asset[number])
            a_learn_asset.append(self.a_memory_asset[number])
            r_learn_asset.append(self.r_memory_asset[number])
            sn_learn_asset.append(self.sn_memory_asset[number])
            correlation_asset.append(self.correlation_matrix[number])
            correlation_asset_n.append(self.correlation_n_matrix[number])

            s_learn_market.append(self.s_memory_market[number])
            a_learn_market.append(self.a_memory_market[number])
            r_learn_market.append(self.r_memory_market[number])
            sn_learn_market.append(self.sn_memory_market[number])
            roh_bar_market.append(self.roh_bars[number])
        self.critic_learn_time = self.critic_learn_time + 1
        # update the asset unit
        # 除了correlation以外都是tensor correlation是np.array 直接从make_correlation_information返回即可
        for bs, ba, br, bs_, correlation, correlation_n in zip(
                s_learn_asset, a_learn_asset, r_learn_asset, sn_learn_asset,
                correlation_asset, correlation_asset_n):
            # update actor
            a = self.act(bs, correlation)
            q = self.cri(bs, correlation, a)
            a_loss = -torch.mean(q)
            self.act_optimizer.zero_grad()
            a_loss.backward(retain_graph=True)
            self.act_optimizer.step()
            # update critic
            a_ = self.act(bs_, correlation_n)
            q_ = self.cri(bs_, correlation_n, a_.detach())
            q_target = br + self.gamma * q_
            q_eval = self.cri(bs, correlation, ba.detach())
            # print(q_eval)
            # print(q_target)
            td_error = self.loss(q_target.detach(), q_eval)
            # print(td_error)
            self.cri_optimizer.zero_grad()
            td_error.backward()
            self.cri_optimizer.step()
        # update the asset unit
        # 除了correlation以外都是tensor correlation是np.array 直接从make_correlation_information返回即可
        loss_market = 0
        for s, br, roh_bar in zip(s_learn_market, r_learn_asset,
                                  roh_bar_market):
            output_market = self.market(s)
            normal = Normal(output_market[0], output_market[1])
            b_prob = -normal.log_prob(roh_bar)

            loss_market += br * b_prob

        self.market_optimizer.zero_grad()
        loss_market.backward()
        self.market_optimizer.step()