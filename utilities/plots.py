import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from environment.state_handling import get_prototype, get_storage_path


def plot_absolute_results(rewards, steps, num_episodes, episode_delimiter):
    plt.subplot(211)  # 2 rows for subplots, 1 column, idx 1
    plt.plot(range(1, num_episodes + 1), rewards)
    plt.ylabel("Rewards")

    plt.subplot(212)
    plt.plot(range(1, num_episodes + 1), steps)
    plt.ylabel("Steps")

    plt.xlabel("Episodes")

    run_info = "p{}-{}e-{}s".format(get_prototype(), num_episodes, episode_delimiter)
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    fig_file = os.path.join(get_storage_path(), "results-fig={}={}.png".format(timestamp, run_info))
    plt.savefig(fig_file)
    return fig_file


def plot_average_results(rewards, steps, num_episodes, episode_delimiter):
    avg_rewards = __exp_moving_average(np.array(rewards), 10 / num_episodes)
    # avg_rewards = __exp_moving_average(np.array(rewards), 1/100)
    # avg_rewards = __exp_moving_average(np.array(rewards), 1/1000)
    avg_steps = __exp_moving_average(np.array(steps), 10 / num_episodes)
    # avg_steps = __exp_moving_average(np.array(steps), 1/100)
    # avg_steps = __exp_moving_average(np.array(steps), 1/1000)
    return __plot_combined_results(rewards, avg_rewards, steps, avg_steps, num_episodes, episode_delimiter)


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


def __plot_combined_results(rewards, avg_rewards, steps, avg_steps, num_episodes, episode_delimiter):
    plt.subplot(211)  # 2 rows for subplots, 1 column, idx 1
    plt.plot(range(1, num_episodes + 1), rewards, color="blue")
    plt.plot(range(1, num_episodes + 1), avg_rewards, color="red")
    plt.ylabel("Rewards")
    plt.legend(["Absolute", "Average"])

    plt.subplot(212)
    plt.plot(range(1, num_episodes + 1), steps, color="blue")
    plt.plot(range(1, num_episodes + 1), avg_steps, color="red")
    plt.ylabel("Steps")
    plt.legend(["Absolute", "Average"])

    plt.xlabel("Episodes")

    run_info = "p{}-{}e-{}s".format(get_prototype(), num_episodes, episode_delimiter)
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    fig_file = os.path.join(get_storage_path(), "results-fig={}={}.png".format(timestamp, run_info))
    plt.savefig(fig_file)
    return fig_file
