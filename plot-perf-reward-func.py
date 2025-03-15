import math
import os

from matplotlib import use as use_plot_backend
import matplotlib.pyplot as plt
import numpy as np
from environment.state_handling import initialize_storage, cleanup_storage, get_storage_path

def posold(r):
    h = +0
    return list(map(lambda v: 10 * math.log10(v + 1) + abs(h), r))  # Old version Reward function when hidden (log = log_e = ln)

def pos(r):
    h = +0
    return list(map(lambda v: 100 * math.log10(v*0.01 + 1) + abs(h), r))  # Current version Reward function when hidden (log = log_e = ln)

def neg(r):
    d = -20
    return list(map(lambda v: (d/max(v, 1)) - abs(d), r))   # Current version Reward function when detected (-d/r - |d|)

try:
    initialize_storage()
    use_plot_backend("template")

    x_lim = 500
    description = "posold-log10-pos-scaledln-neg-1overX-h{}-d{}-xlim-{}".format("+0", "-20", x_lim)

    x = np.linspace(0, x_lim, num=500)

    plt.plot(x, posold(x), label="Hidden (Old)", color="red")
    plt.plot(x, pos(x), label="Hidden (Current)", color="blue")
    plt.plot(x, neg(x), label="Detected (Current)", color="green")

    plt.ylabel("Reward")
    plt.xlabel("Encryption Rate")
    plt.legend()
    plt.grid()
    plt.title("Performance Reward Functions")
    plt.xlim(0, x_lim)

    fig_file = os.path.join(get_storage_path(), "reward-func={}.png".format(description))
    plt.savefig(fig_file)
finally:
    cleanup_storage()
