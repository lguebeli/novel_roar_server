import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as use_plot_backend
from environment.state_handling import initialize_storage, cleanup_storage, get_storage_path

def relu(x):
    return np.maximum(0, x)         # ReLU activation function

def logistic(x):
    return 1 / (1 + np.exp(-x))     # Logistic (Sigmoid) activation function

def silu(x):
    return x / (1 + np.exp(-x))     # SiLU activation function.

def tanh(x):
    return np.tanh(x)               # Tanh activation function.

try:
    initialize_storage()
    use_plot_backend("template")

    x_lim = 4
    description = "ReLU-SiLU-Tanh-xlim-{}".format(x_lim)

    x = np.linspace(-x_lim, x_lim, num=500)

    # Plot all four activation functions with distinct colors
    plt.plot(x, relu(x), label="ReLU", color="purple")
    plt.plot(x, logistic(x), label="Logistic", color="green")
    plt.plot(x, silu(x), label="SiLU", color="red")
    plt.plot(x, tanh(x), label="Tanh", color="blue")

    plt.legend()
    plt.xlim(-x_lim, x_lim)
    plt.grid()
    plt.title("Activation Functions: ReLU, SiLU, Tanh")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    fig_file = os.path.join(get_storage_path(), "activation-func={}.png".format(description))
    plt.savefig(fig_file)
finally:
    cleanup_storage()