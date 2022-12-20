from datetime import datetime
import matplotlib.pyplot as plt
import os

from environment.state_handling import get_prototype, get_storage_path


def plot_results(rewards, steps, num_episodes, episode_delimiter):
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
