import torch
from torch import nn
import numpy as np
from torch.distributions import Normal


def generate_portfolio(scores=torch.sigmoid(torch.randn(29, 1)), quantile=0.5):
    scores = scores.squeeze()
    sorted_score, indices = torch.sort(scores, descending=True)
    length = len(scores)
    rank_hold = int(quantile * length)
    value_hold = sorted_score[rank_hold - 1]
    good_portfolio = []
    good_scores = []
    bad_portfolio = []
    bad_scores = []
    for i in range(length):
        score = scores[i]
        if score <= value_hold:
            bad_portfolio.append(i)
            bad_scores.append(score)
        else:
            good_portfolio.append(i)
            good_scores.append(score)
    final_portfollio = [0] * length
    good_portion = np.exp(good_scores) / np.sum(
        np.exp(good_scores)) * (quantile)
    bad_portion = -np.exp(1 - np.array(bad_scores)) / np.sum(
        np.exp(1 - np.array(bad_scores))) * (1 - quantile)
    for i in range(length):
        if i in bad_portfolio:
            index = bad_portfolio.index(i)
            final_portfollio[i] = bad_portion[index]
        else:
            index = good_portfolio.index(i)
            final_portfollio[i] = good_portion[index]

    return final_portfollio


def generate_rho(mean: torch.tensor, std: torch.tensor):
    normal = Normal(mean, std)
    result = normal.sample()
    return result


if __name__ == "__main__":
    print(generate_portfolio())
    print(sum(generate_portfolio()))
    print(sum(np.abs(generate_portfolio())))
