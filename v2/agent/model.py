import numpy as np

from environment.state_handling import get_num_configs

FP_DIMS = 103
HIDDEN_NEURONS = 200
NUM_CONFIGS = get_num_configs()


class Model(object):
    def __init__(self, epsilon, learn_rate, decay_rate):
        # Initialize random seed for reproducibility
        np.random.seed(42)

        # Initialize hyperparams
        self.eps_decay = decay_rate
        self.epsilon_0 = epsilon
        self.epsilon_forward = epsilon
        self.learn_rate = learn_rate

        # Initialize neural network
        self.allowed_actions = np.asarray(range(NUM_CONFIGS))

        num_input = FP_DIMS  # Input size
        num_hidden = HIDDEN_NEURONS  # Hidden neurons
        num_output = NUM_CONFIGS  # Output size

        self.weights1 = np.random.uniform(0, 1, (num_hidden, num_input))
        self.weights2 = np.random.uniform(0, 1, (num_output, num_hidden))
        # TODO: v3 - xavier weight init

        self.bias_weights1 = np.zeros((num_hidden,))
        self.bias_weights2 = np.zeros((num_output,))

        self.x1 = np.ones((num_hidden,))
        self.q = np.ones((num_output,))

    def forward(self, inputs, step_n):
        # Forward pass through neural network to compute Q-values
        h1 = np.dot(self.weights1, inputs) + self.bias_weights1
        self.x1 = 1 / (1 + np.exp(-h1))  # sigmoid logistic activation
        h2 = np.dot(self.weights2, self.x1) + self.bias_weights2
        self.q = 1 / (1 + np.exp(-h2))  # sigmoid logistic activation, x2
        # TODO: v3 - switch to ReLU - https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

        # Choose action based on epsilon-greedy policy
        possible_a = self.allowed_actions
        q_a = self.q[possible_a]

        if np.random.random() < self.epsilon_forward:  # explore randomly
            # print("MODEL: random action")
            sel_a = possible_a[np.random.randint(possible_a.size)]
        else:  # exploit greedily
            # print("MODEL: greedy action")
            sel_a = possible_a[np.argmax(q_a)]

        # TODO: originally decay every episode! n = episode number
        #  v3 - put in controller
        # self.epsilon_forward = self.epsilon_0 / (1 + self.eps_decay * step_n)  # decay epsilon
        self.epsilon_forward = self.epsilon_0
        return sel_a, self.q

    def backward(self, fingerprint, q_err):
        # Backpropagation of error through neural network
        delta2 = self.q * (1 - self.q) * q_err
        delta_weights2 = np.outer(delta2, self.x1)
        delta1 = self.x1 * (1 - self.x1) * np.dot(self.weights2.T, delta2)
        delta_weights1 = np.outer(delta1, fingerprint)

        self.weights1 += self.learn_rate * delta_weights1
        self.weights2 += self.learn_rate * delta_weights2
        self.bias_weights1 += self.learn_rate * delta1
        self.bias_weights2 += self.learn_rate * delta2
