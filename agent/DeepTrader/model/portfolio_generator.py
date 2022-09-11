import torch
from torch import nn
import numpy as np
from torch.distributions import Normal


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


if __name__ == "__main__":
    print(generate_portfolio())
    print(sum(generate_portfolio()))
    print(sum(np.abs(generate_portfolio())))
