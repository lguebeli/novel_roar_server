import os

import matplotlib.pyplot as plt
import numpy as np

from environment.state_handling import get_storage_path


def plot_absolute_results(rewards, steps, num_episodes, description):
    plt.subplot(211)  # 2 rows for subplots, 1 column, idx 1
    plt.plot(range(1, num_episodes + 1), rewards)
    plt.ylabel("Rewards")

    plt.subplot(212)
    plt.plot(range(1, num_episodes + 1), steps)
    plt.ylabel("Steps")

    plt.xlabel("Episodes")

    fig_file = os.path.join(get_storage_path(), "results-fig={}.png".format(description))
    plt.savefig(fig_file)
    return fig_file


def plot_average_results(avg_rewards, steps, num_episodes, description):
    ema_rewards = __exp_moving_average(np.array(avg_rewards), 10 / num_episodes)
    ema_steps = __exp_moving_average(np.array(steps), 10 / num_episodes)
    return __plot_combined_results(avg_rewards, ema_rewards, steps, ema_steps, num_episodes, description)


def __exp_moving_average(data, alpha):
    # https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def __plot_combined_results(avg_rewards, ema_rewards, steps, ema_steps, num_episodes, description):
    fig, (ax1, ax2) = plt.subplots(2, 1)  # 2 rows for subplots, 1 column

    # ==============================
    # PLOT REWARDS
    # ==============================

    ax1.scatter(range(1, num_episodes + 1), avg_rewards, s=5, color="blue")
    ax1.plot(range(1, num_episodes + 1), ema_rewards, color="red")
    ax1.set_ylabel("Average Rewards")

    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.legend(["Abs", "EMA"])

    # ==============================
    # PLOT STEPS
    # ==============================

    ax2.scatter(range(1, num_episodes + 1), steps, s=5, color="blue")
    ax2.plot(range(1, num_episodes + 1), ema_steps, color="red")
    ax2.set_ylabel("Steps")

    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.legend(["Abs", "EMA"])

    # ==============================
    # FINALIZE AND SAVE FIGURE
    # ==============================

    ax2.set_xlabel("Episodes")
    fig.align_ylabels()

    fig_file = os.path.join(get_storage_path(), "results-fig={}.png".format(description))
    plt.savefig(fig_file)
    return fig_file
