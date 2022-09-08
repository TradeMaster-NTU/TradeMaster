import sys

sys.path.append(".")
from agent.DeepTrader.model.model import IN, Chomp1d, TemporalBlock, TemporalConvNet, SA, GCN, IN, market_scoring, asset_scoring, asset_scoring_value
import torch
from env.PM.portfolio_for_deeptrader import *
from env.PM.portfolio_for_deeptrader import Tradingenv
from agent.DeepTrader.data.market_information import *
from agent.DeepTrader.model.portfolio_generator import generate_portfolio, generate_rho

a = Tradingenv(vars(args))
state = a.reset()
corr_matrix = make_correlation_information(a.data)
market_information = make_market_information(
    a.data,
    technical_indicator=[
        "high", "low", "open", "close", "adjcp", "zopen", "zhigh", "zlow",
        "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25", "zd_30"
    ])
input_asset = torch.from_numpy(state).to(torch.float32)
print(input_asset.shape)
net_asset = asset_scoring(N=29,
                          K_l=10,
                          num_inputs=16,
                          num_channels=[12, 12, 12])
output_asset = net_asset(input_asset, A=corr_matrix)
net = asset_scoring_value(N=29,
                          K_l=10,
                          num_inputs=16,
                          num_channels=[12, 12, 12])
print(net(input_asset, A=corr_matrix, action=output_asset).shape)
input_market = torch.from_numpy(market_information).unsqueeze(0).to(
    torch.float32)
output_market = market_scoring(16)(input_market)
roh_bar = generate_rho(output_market[0], output_market[1])
print(roh_bar)
print(generate_portfolio(output_asset, roh_bar))
action = generate_portfolio(output_asset, roh_bar)
action = action.numpy()
state_ = a.step(action)
print(state.shape)
# input_market = torch.from_numpy(market_information).to(
#     torch.float32).unsqueeze(0)
# output_market = market_scoring(16)(input_market)
# rhou_bar = generate_rho(output_market)
# net = market_scoring(16)
# print(
#     torch.from_numpy(market_information).to(torch.float32).unsqueeze(0).shape)
# result = net(
#     torch.from_numpy(market_information).to(torch.float32).unsqueeze(0))
# print(result.shape)
# net = TemporalConvNet(16, [12, 12, 12, 12])
# print(net(torch.from_numpy(state).to(torch.float32)).transpose(0, 1).shape)
# C = 12
# N = 29
# K_l = 10
# a = net(torch.from_numpy(state).to(torch.float32)).transpose(0, 1)
# net = SA(C, N, K_l)
# S_l = net(a)
# print(S_l.shape)
# gcn = GCN(K_l=10)
# Z_l = gcn(np.random.random(size=(29, 29)), a)
# print(Z_l.shape)
# inet = IN(29)
# output = inet(S_l, Z_l, a)
# print(output.shape)
# dataframe = a.data
# a = Tradingenv(vars(args))
# state = make_market_information(a.data, [
#     "high", "low", "open", "close", "adjcp", "zopen", "zhigh", "zlow",
#     "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25", "zd_30"
# ]).values
# net = market_scoring(16)
# print(torch.from_numpy(state).unsqueeze(0).to(torch.float32).shape)
# h_k = net(torch.from_numpy(state).unsqueeze(0).to(torch.float32))
# print(h_k.shape)
# # W_1 = torch.randn(K_l)
# # W_2 = torch.randn(C, K_l)
# # W_3 = torch.randn(C)
# # V_s = torch.randn(N, N)
# # b_s = torch.randn(N, N)

# # S_l = torch.matmul(torch.matmul(torch.matmul(a, W_1).T, W_2),
# #                    torch.matmul(W_3, a.transpose(0, 1)).T)
# # print(S_l.shape)
