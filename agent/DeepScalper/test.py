# import argparse
# import yaml
# # TODO df略有不同 需要有一列vadility来进行计算
# parser = argparse.ArgumentParser()
# # from stable_baselines3.common.env_checker import check_envs

# parser.add_argument(
#     "--df_path",
#     type=str,
#     default="./experiment_result/data/s_test.csv",
#     help="the path for the downloaded data to generate the environment")
# parser.add_argument("--initial_amount",
#                     type=int,
#                     default=100000,
#                     help="the initial amount of money for trading")
# parser.add_argument("--transaction_cost_pct",
#                     type=float,
#                     default=0.001,
#                     help="the transcation cost for us to ")
# parser.add_argument("--tech_indicator_list",
#                     type=list,
#                     default=[
#                         "high", "low", "open", "close", "adjcp", "zopen",
#                         "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10",
#                         "zd_15", "zd_20", "zd_25", "zd_30"
#                     ],
#                     help="the name of the features to predict the label")
# parser.add_argument(
#     "--forward_num_day",
#     type=int,
#     default=5,
#     help="the number of day to calculate the long period of profit ",
# )
# parser.add_argument(
#     "--backward_num_day",
#     type=int,
#     default=5,
#     help="the number of day to calculate the variance of the assets ",
# )
# parser.add_argument(
#     "--future_weights",
#     type=float,
#     default=0.2,
#     help="the rewards here defined is a little different ",
# )
# parser.add_argument(
#     "--max_volume",
#     type=int,
#     default=1,
#     help="the max volume of bitcoin you can buy at one time ",
# )
# args = parser.parse_args()
# config = vars(args)
# print(config)

# def save_dict_to_yaml(dict_value: dict, save_path: str):
#     """dict保存为yaml"""
#     with open(save_path, 'w') as file:
#         file.write(yaml.dump(dict_value, allow_unicode=True))

# save_dict_to_yaml(
#     config,
#     "/home/sunshuo/qml/TradeMaster_reframe/input_config/env/AT/AT/test.yml")

# def read_yaml_to_dict(yaml_path: str, ):
#     with open(yaml_path) as file:
#         dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
#         return dict_value

# print(read_yaml_to_dict("input_config/env/AT/AT/test.yml"))
a = [0, 1, 2, 3, 4, 5, 6]
print(a[0:len(a):2])
print(0.1 < 0.2 and 0.1 >= 0)
print(sum([[0.1, 0.2], [0.3]]))
