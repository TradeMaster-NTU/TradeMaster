import numpy as np
import torch
import sys

sys.path.append(".")
from agent.iRDPG.RL.model import RNN, Actor, Critic


class Agent(object):
    # 就是DPG 只不过加上一开始是demeostration的action
    #外加BC
    def __init__(self, input_size, seq_len, num_rnn_layer, hidden_rnn,
                 rnn_mode, init_w, hidden_fc1, hidden_fc2, hidden_fc3,
                 Reward_max_clip, discount):
        nb_actions = 2
        # if args.seed > 0:
        #     self.seed(args.seed)

        ##### Create RNN Layer #####
        self.rnn = RNN(input_size, seq_len, num_rnn_layer, hidden_rnn,
                       rnn_mode)
        self.rnn_target = RNN(input_size, seq_len, num_rnn_layer, hidden_rnn,
                              rnn_mode)
        ##### Create Actor Network #####
        self.actor = Actor(init_w, hidden_rnn, hidden_fc1, hidden_fc2,
                           hidden_fc3)
        self.actor_target = Actor(init_w, hidden_rnn, hidden_fc1, hidden_fc2,
                                  hidden_fc3)
        ##### Create Critic Network #####
        self.critic = Critic(init_w, input_size, seq_len, num_rnn_layer,
                             hidden_rnn, hidden_fc1, hidden_fc2, hidden_fc3,
                             Reward_max_clip, discount)
        self.critic_target = Critic(init_w, input_size, seq_len, num_rnn_layer,
                                    hidden_rnn, hidden_fc1, hidden_fc2,
                                    hidden_fc3, Reward_max_clip, discount)
