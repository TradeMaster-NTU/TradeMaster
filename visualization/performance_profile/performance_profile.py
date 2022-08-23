import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import seaborn as sns

sns.set_style("white")
import matplotlib.patches as mpatches
import collections
import os


def make_distribution_plot(dict_algorithm, algorithms, reps, xlabel, dic,
                           colors):
    color_idxs = [0, 1, 2, 3, 4, 5, 6, 7]
    ATARI_100K_COLOR_DICT = dict(
        zip(algorithms, [colors[idx] for idx in color_idxs]))

    score_dict = {key: dict_algorithm[key][:] for key in algorithms}
    ATARI_100K_TAU = np.linspace(-1, 100, 1000)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, ATARI_100K_TAU, reps=reps)
    fig, ax = plt.subplots(ncols=1, figsize=(8.0, 4.0))
    plot_utils.plot_performance_profiles(
        score_distributions,
        ATARI_100K_TAU,
        performance_profile_cis=score_distributions_cis,
        colors=ATARI_100K_COLOR_DICT,
        xlabel=xlabel,
        labelsize='x-large',
        ax=ax)
    ax.axhline(0.5, ls='--', color='k', alpha=0.4)
    fake_patches = [
        mpatches.Patch(color=ATARI_100K_COLOR_DICT[alg], alpha=0.75)
        for alg in algorithms
    ]
    legend = fig.legend(loc='upper left',
                        ncol=int(len(algorithms) / 2),
                        fontsize=15,
                        bbox_to_anchor=(0, 1, 1, 0.1))
    plt.savefig(dic, bbox_inches='tight')
