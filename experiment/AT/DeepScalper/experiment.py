import argparse
import sys

sys.path.append(".")
from agent.DeepScalper.dqn import *


def build_agent():
    a = DQN(args)
    return a


def experiment(a: DQN):
    a.train_with_valid()
    a.test()


def main():
    a = build_agent()
    experiment(a)


if __name__ == "__main__":
    main()