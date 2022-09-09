import sys

sys.path.append("./")
from data.download_data import Dataconfig
from agent.ClassicRL.SOTA import trader as Agent
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",
                    type=str,
                    default="./data/data/",
                    help="the path for storing the downloaded data")
#where we store the dataset
parser.add_argument(
    "--output_config_path",
    type=str,
    default="./output_config/data",
    help="the path for storing the generated config file for data")
# where we store the config file
parser.add_argument(
    "--dataset",
    choices=["dj30", "sz50", "acl18", "futures", "crypto", "exchange"],
    default="dj30",
    help="the name of the dataset",
)
parser.add_argument("--split_proportion",
                    type=list,
                    default=[0.8, 0.1, 0.1],
                    help="the split proportion for train, valid and test")
parser.add_argument(
    "--generate_config",
    type=bool,
    default=True,
    help=
    "determine whether to generate a yaml file to memorize the train valid and test'data's dict"
)
parser.add_argument(
    "--input_config",
    type=bool,
    default=False,
    help=
    "determine whether to use a yaml file as the overall input of the Dataconfig, this is needed when have other format of dataset"
)

parser.add_argument(
    "--input_config_path",
    type=str,
    default="input_config/data/custom.yml",
    help=
    "determine the location of a yaml file used to initialize the Dataconfig Class"
)

parser.add_argument(
    "--env_name",
    choices=["portfolio"],
    default="portfolio",
    help="the name of TradingEnv ",
)
parser.add_argument(
    "--dict_trained_model",
    default="experiment_result/SOTA/trained_model/",
    help="the dict of the trained model ",
)

parser.add_argument(
    "--train_env_config_dict",
    default="input_config/env/portfolio/portfolio/train.yml",
    help="the dict of the train config of TradingEnv ",
)

parser.add_argument(
    "--valid_env_config_dict",
    default="input_config/env/portfolio/portfolio/valid.yml",
    help="the dict of the valid config of TradingEnv ",
)

parser.add_argument(
    "--test_env_config_dict",
    default="input_config/env/portfolio/portfolio/test.yml",
    help="the dict of the test config of TradingEnv ",
)

parser.add_argument(
    "--name_of_algorithms",
    choices=["PPO", "A2C", "SAC", "TD3", "PG", "DDPG"],
    type=str,
    default="A2C",
    help="name_of_algorithms ",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="the number of training epoch",
)

parser.add_argument(
    "--random_seed",
    type=int,
    default=12345,
    help="the number of training epoch",
)

parser.add_argument(
    "--model_config_dict",
    type=str,
    default="input_config/agent/SOTA/A2C.yml",
    help="the dict of the model_config file",
)

parser.add_argument(
    "--result_dict",
    type=str,
    default="experiment_result/SOTA/test_result/",
    help="the dict of the result of the test",
)

args = parser.parse_args()


def get_data(args):
    Dataconfig(args)


def get_agent(args):
    agent = Agent(args)
    return agent


def experiment(agent: Agent):
    agent.train_with_valid()
    agent.test()


def main(args):
    Dataconfig(args)
    agent = Agent(args)
    experiment(agent)


if __name__ == '__main__':
    main(args)
