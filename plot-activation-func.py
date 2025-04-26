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

def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)  # Leaky ReLU activation function

try:
    initialize_storage()
    use_plot_backend("template")

    x_lim = 4
    description = "ReLu-Log-SiLU-LeReLU-{}".format(x_lim)

    x = np.linspace(-x_lim, x_lim, num=500)

    # Plot all four activation functions with distinct colors
    plt.plot(x, relu(x), label="ReLU", color="red")
    plt.plot(x, logistic(x), label="Logistic", color="green")
    plt.plot(x, silu(x), label="SiLU", color="red")
    plt.plot(x, leaky_relu(x), label="Leaky ReLU", color="blue")

    plt.legend()
    plt.xlim(-x_lim, x_lim)
    plt.grid()
    plt.title("Activation Functions: ReLU, Logarithmic, SiLU, LeakyReLu")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    fig_file = os.path.join(get_storage_path(), "activation-func={}.png".format(description))
    plt.savefig(fig_file)
finally:
    cleanup_storage()