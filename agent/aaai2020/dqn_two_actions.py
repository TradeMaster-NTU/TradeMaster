import argparse
import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append("./")
from agent.aaai2020.tianshou.data import ReplayBuffer, StockCollector_TwoActions
from agent.aaai2020.tianshou.env import StockVectorEnv_TwoActions
from agent.aaai2020.tianshou.env.stock import create_stock_environment
from agent.aaai2020.tianshou.policy import DQNPolicyTwoActions
from agent.aaai2020.tianshou.trainer import offpolicy_trainer


class stock_DQN_TwoActions(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 inventory_shape,
                 layer_num=1,
                 device='cpu'):
        super(stock_DQN_TwoActions, self).__init__()
        self.device = device

        self.action_shape = action_shape
        self.inventory_shape = inventory_shape

        linear_input_size = int(np.prod(state_shape))
        self.device = device
        self.model = [nn.Linear(linear_input_size, 128), nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, action_shape * inventory_shape)]

        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='./data/600030')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.5)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device',
                        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    env = create_stock_environment('{}_valid.csv'.format(args.task))
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # train_envs = gym.make(args.task)
    train_envs = StockVectorEnv_TwoActions([
        lambda: create_stock_environment('{}_train.csv'.format(args.task))
        for _ in range(args.training_num)
    ],
                                           mode='train')
    # test_envs = gym.make(args.task)
    test_envs = StockVectorEnv_TwoActions([
        lambda: create_stock_environment('{}_valid.csv'.format(args.task))
        for _ in range(args.test_num)
    ],
                                          mode='test')
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = stock_DQN_TwoActions(args.state_shape, args.action_shape, 10, 1,
                               args.device)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicyTwoActions(
        net,
        optim,
        args.gamma,
        args.n_step,
        use_target_network=args.target_update_freq > 0,
        target_update_freq=args.target_update_freq)
    # collector
    train_collector = StockCollector_TwoActions(policy, train_envs,
                                                ReplayBuffer(args.buffer_size))
    test_collector = StockCollector_TwoActions(policy, test_envs)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * 4)
    print(len(train_collector.buffer))
    # log
    writer = SummaryWriter(args.logdir + '/' + 'dqn')

    def stop_fn(x):
        if env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        else:
            return False

    def train_fn(x):
        policy.set_eps(max(args.eps_train * (10000 - x) * 10000, 0.1))

    def test_fn(x):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(policy,
                               train_collector,
                               test_collector,
                               args.epoch,
                               args.step_per_epoch,
                               args.collect_per_step,
                               args.test_num,
                               args.batch_size,
                               train_fn=train_fn,
                               test_fn=test_fn,
                               stop_fn=stop_fn,
                               writer=writer,
                               task=args.task)

    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = create_stock_environment('{}_valid.csv'.format(args.task))
        collector = StockCollector_TwoActions(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_dqn(get_args())
