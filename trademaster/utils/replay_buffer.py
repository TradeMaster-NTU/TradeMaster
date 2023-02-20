import os
import torch
import numpy as np
import numpy.random as rd
from torch import Tensor
from collections import deque, namedtuple
import random
class ReplayBuffer:  # for off-policy, vectorized env
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 device: torch.device,
                 num_envs: int = 1,
                 if_use_per: bool = False):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_envs = num_envs
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = torch.empty((max_size, num_envs, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, num_envs, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, num_envs), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, num_envs), dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per
        if if_use_per:
            self.per_tree = BinarySearchTree(max_size * num_envs)
            self.sample = self.sample_per
    def update(self, items: list):
        self.add_item = items
        states, actions, rewards, undones = items
        assert states.shape[1:] == (self.num_envs, self.state_dim)
        assert actions.shape[1:] == (self.num_envs, self.action_dim)
        assert rewards.shape[1:] == (self.num_envs,)
        assert undones.shape[1:] == (self.num_envs,)
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> list:
        sample_len = self.cur_size - 1

        ids = torch.randint(sample_len * self.num_envs, size=(batch_size,), requires_grad=False)
        ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        return (self.states[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],)  # next_state

    def sample_per(self, batch_size: int) -> list:
        beg = -self.max_size
        end = (self.cur_size - self.max_size) if (self.cur_size < self.max_size) else -1

        ids, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)
        is_weights = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)  # important sampling weights

        ids0 = torch.fmod(ids, self.cur_size)  # ids % sample_len
        ids1 = torch.div(ids, self.cur_size, rounding_mode='floor')  # ids // sample_len

        return (self.states[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],  # next_state
                is_weights)

    def td_error_update(self, td_error: Tensor):
        self.per_tree.td_error_update(td_error)

    def save_or_load_history(self, cwd: str, if_save: bool):
        item_names = (
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        )

        if if_save:
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pt"
                print(f"| {self.__class__.__name__}: Save {file_path}")
                torch.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pt") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pt"
                print(f"| {self.__class__.__name__}: Load {file_path}")
                buf_item = torch.load(file_path)

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = max_sizes[0]


class BinarySearchTree:
    """Binary Search Tree for PER
    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len
        self.prob_ary = np.zeros((memo_len - 1) + memo_len)  # parent_nodes_num + leaf_nodes_num
        self.max_capacity = len(self.prob_ary)
        self.cur_capacity = self.memo_len - 1  # pointer
        self.indices = None
        self.depth = int(np.log2(self.max_capacity))

        """
        PER.  Prioritized Experience Replay. Section 4
        alpha, beta = 0.7, 0.5 for rank-based variant
        alpha, beta = 0.6, 0.4 for proportional variant
        """
        self.per_alpha = 0.6  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)

    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1
        if self.cur_capacity == tree_id:
            self.cur_capacity += 1

        delta = prob - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = prob

        while tree_id != 0:  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):  # 10 is max_prob
        ids = data_ids + self.memo_len - 1
        self.cur_capacity += (ids >= self.cur_capacity).sum()

        upper_step = self.depth - 1
        self.prob_ary[ids] = prob  # here, ids means the indices of given children (maybe the right ones or left ones)
        p_ids = (ids - 1) // 2

        while upper_step:  # propagate the change through tree
            ids = p_ids * 2 + 1  # in this while loop, ids means the indices of the left children
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
        # because we take depth-1 upper steps, ps_tree[0] need to be updated alone

    def get_leaf_id(self, v):
        """Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx
        return min(leaf_idx, self.cur_capacity - 2)  # leaf_idx

    def get_indices_is_weights(self, batch_size, start=None, end=None):
        self.per_beta = min(1., self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.prob_ary[0] / batch_size)

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[start:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return self.indices, is_weights

    def td_error_update(self, td_error):  # td_error = (q-q).detach_().abs()
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)
        
        
#TODO 目前自己独有的replaybuffer 不具有horizon length 但是可以存储info 
class ReplayBufferHFT:  # for off-policy, vectorized env, containing info from the env
    def __init__(self,
                 buffer_size:int,
                 batch_size:int,
                 device:torch.device,
                 seed,
                 gamma,
                 n_step,
                 parallel_env=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=[
                                         "state", "info", "action", "reward",
                                         "next_state", "done", "next_info"
                                     ])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [
            deque(maxlen=self.n_step) for i in range(parallel_env)
        ]
        self.iter_ = 0

    def add(
        self,
        state,
        info: dict,
        action,
        reward,
        next_state,
        next_info: dict,
        done,
    ):
        """Add a new experience to memory."""
        # if we want to have multi core
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append(
            (state, info, action, reward, next_state, next_info, done))

        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, info, action, reward, next_state, next_info, done = self.calc_multistep_return(
                self.n_step_buffer[self.iter_])
            e = self.experience(state, info, action, reward, next_state, done,
                                next_info)
            self.memory.append(e)
        self.iter_ += 1

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][3]
        self.info_key = n_step_buffer[0][5].keys()

        return n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][
            2], Return, n_step_buffer[-1][4], n_step_buffer[-1][
                5], n_step_buffer[0][6]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences
                      if e is not None])).float().to(self.device)
        infos = dict()
        for key in self.info_key:
            infos[key] = torch.from_numpy(
                np.stack([e.info[key] for e in experiences if e is not None
                          ]).astype(np.uint8)).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences
                      if e is not None])).float().to(self.device)
        next_infos = dict()
        for key in self.info_key:
            next_infos[key] = torch.from_numpy(
                np.stack([
                    e.next_info[key] for e in experiences if e is not None
                ]).astype(np.uint8)).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(np.uint8)).float().to(self.device)

        return (states, infos, actions, rewards, next_states, next_infos,
                dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

