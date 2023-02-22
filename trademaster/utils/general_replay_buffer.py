import os
import torch
import numpy as np
import numpy.random as rd
from torch import Tensor
from collections import namedtuple, OrderedDict

Transition = namedtuple("Transition", ['state', 'action', 'reward', 'undone','next_state'])

class GeneralReplayBuffer:
    def __init__(self,
                 transition: namedtuple,
                 shapes: dict,
                 max_size: int,
                 num_seqs: int,
                 device: torch.device
                 ):

        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = device

        # initialize
        self.transition = transition
        self.names = self.transition._fields
        self.shapes = shapes
        self.storage = OrderedDict()
        for name in self.names:
            assert name in self.shapes
            # (max_size, num_seqs, dim1, dim2, ...)
            self.storage[name] = torch.empty(self.shapes[name], dtype=torch.float32, device=self.device)

    def update(self, items: namedtuple):
        # check shape
        for name in self.names:
            assert name in self.storage
            assert getattr(items, name).shape[1:] == self.storage[name].shape[1:]

        # add size
        self.add_size = getattr(items, self.names[0]).shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            for name in self.names:
                self.storage[name][p0:p1], self.storage[name][0:p] = \
                    getattr(items, name)[:p2], getattr(items, name)[-p:]
        else:
            for name in self.names:
                self.storage[name][self.p:p] = getattr(items, name)
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def clear(self):
        for name in self.names:
            assert name in self.shapes
            # (max_size, num_seqs, dim1, dim2, ...)
            self.storage[name] = torch.empty(self.shapes[name], dtype=torch.float32, device=self.device)
    def sample(self, batch_size: int) -> namedtuple:
        sample_len = self.cur_size - 1

        ids = torch.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False)
        ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        sample_data = OrderedDict()
        for name in self.names:
            sample_data[name] = self.storage[name][ids0, ids1]
        return self.transition(**sample_data)

    def save_or_load_history(self, cwd: str, if_save: bool):
        if if_save:
            for name, item in self.storage.items():
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pt"
                print(f"| {self.__class__.__name__}: Save {file_path}")
                torch.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pt") for name, item in self.storage.items()]):
            max_sizes = []
            for name, item in self.storage.items():
                file_path = f"{cwd}/replay_buffer_{name}.pt"
                print(f"| {self.__class__.__name__}: Load {file_path}")
                buf_item = torch.load(file_path)

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = max_sizes[0]