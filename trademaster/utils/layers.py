import torch
import torch.nn as nn

def get_optim_param(optimizer: torch.optim) -> list:
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list

def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)

def build_conv2d(dims: [int], kernel_size: [(int, int)]) -> nn.Sequential:
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Conv2d(in_channels=dims[i],
                                   out_channels=dims[i + 1],
                                   kernel_size = (kernel_size[i][0], kernel_size[i][1])), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)