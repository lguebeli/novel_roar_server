import os
import matplotlib.pyplot as plt
import numpy as np
from environment.state_handling import get_storage_path

def plot_cpu_usage(per_episode_resources, description):
    episodes = [r["episode"] for r in per_episode_resources]
    max_cpu_percent = [r["max_cpu"] for r in per_episode_resources]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(6, 6)

    # Plot CPU percentage on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('CPU Usage (%)', color=color)
    ax1.plot(episodes, max_cpu_percent, color=color, label='Max CPU %', linewidth=2.5, zorder=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.set_ylim(30, max(10, max(max_cpu_percent) * 1.1))  # Adjust y-axis to show small percentages clearly

    plt.title('Process CPU Usage per Episode')
    fig.tight_layout()

    plt.savefig(os.path.join(get_storage_path(), f"cpu_usage_{description}.png"))
    plt.close()

def plot_memory_usage(per_episode_resources, description):
    episodes = [r["episode"] for r in per_episode_resources]
    max_memory_used_mb = [r["max_memory_used_mb"] for r in per_episode_resources]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(6, 6)

    # Plot memory usage in MB on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Memory Used (MB)', color=color)
    ax1.plot(episodes, max_memory_used_mb, color=color, label='Max Memory Used (MB)', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.set_ylim(380, max(500, max(max_memory_used_mb) * 1.1))

    plt.title('Process Memory Usage per Episode')
    fig.tight_layout()

    plt.savefig(os.path.join(get_storage_path(), f"memory_usage_{description}.png"))
    plt.close()

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

def plot_average_results(rewards, avg_rewards, steps, num_episodes, description):
    ema_rewards = __exp_moving_average(np.array(rewards), max(10 / num_episodes, 1 / 1000))
    ema_avg_rewards = __exp_moving_average(np.array(avg_rewards), max(10 / num_episodes, 1 / 1000))
    ema_steps = __exp_moving_average(np.array(steps), max(10 / num_episodes, 1 / 1000))
    return __plot_combined_results(rewards, ema_rewards, avg_rewards, ema_avg_rewards, steps, ema_steps, num_episodes, description)

def __exp_moving_average(data, alpha):
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

def __plot_combined_results(rewards, ema_rewards, avg_rewards, ema_avg_rewards, steps, ema_steps, num_episodes, description):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows for subplots, 1 column
    fig.set_size_inches(10, 8)  # width/height in inches
    fig.set_tight_layout(tight=True)

    ax1.scatter(range(1, num_episodes + 1), rewards, s=5, color="blue")
    ax1.plot(range(1, num_episodes + 1), ema_rewards, color="red")
    ax1.set_ylabel("Rewards")
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.legend(["Abs", "EMA"])

    ax2.scatter(range(1, num_episodes + 1), avg_rewards, s=5, color="blue")
    ax2.plot(range(1, num_episodes + 1), ema_avg_rewards, color="red")
    ax2.set_ylabel("Average Rewards")
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.legend(["Abs", "EMA"])

    ax3.scatter(range(1, num_episodes + 1), steps, s=5, color="blue")
    ax3.plot(range(1, num_episodes + 1), ema_steps, color="red")
    ax3.set_ylabel("Steps")
    ax3.xaxis.get_major_locator().set_params(integer=True)
    ax3.yaxis.get_major_locator().set_params(integer=True)
    ax3.legend(["Abs", "EMA"])

    ax3.set_xlabel("Episodes")
    fig.align_ylabels()

    fig_file = os.path.join(get_storage_path(), "results-fig={}.png".format(description))
    plt.savefig(fig_file)
    return fig_file