import numpy as np

from environment.state_handling import get_num_configs

FP_DIMS = 103
HIDDEN_NEURONS = 200
NUM_CONFIGS = get_num_configs()

ELU_ALPHA = 1


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

        self.weights1 = np.random.uniform(0, 1, (num_input, num_hidden))
        self.weights2 = np.random.uniform(0, 1, (num_hidden, num_output))
        # TODO: v3 - xavier weight init

        self.bias_weights1 = np.zeros((num_hidden, 1))
        self.bias_weights2 = np.zeros((num_output, 1))

        self.x1 = np.ones((num_hidden,))
        self.q = np.ones((num_output,))

    def forward(self, inputs, step_n):
        # print("MODEL: inputs", inputs.shape, np.min(inputs), np.argmin(inputs), np.max(inputs), np.argmax(inputs))
        # print("MODEL: w1", self.weights1.shape)

        # Forward pass through neural network to compute Q-values
        h1 = np.dot(self.weights1.T, inputs) + self.bias_weights1
        # print("MODEL: h1 min/max", h1.shape, np.min(h1), np.argmin(h1), np.max(h1), np.argmax(h1))
        self.x1 = h1 * (h1 > 0)  # ReLU activation, x if x > 0 else 0

        # print("MODEL: x1 min/max", self.x1.shape, np.min(self.x1), np.argmin(self.x1), np.max(self.x1),
        #       np.argmax(self.x1))
        # print("MODEL: w2", self.weights2.shape)

        h2 = np.dot(self.weights2.T, self.x1) + self.bias_weights2
        # print("MODEL: h2 min/max", h2.shape, np.min(h2), np.argmin(h2), np.max(h2), np.argmax(h2))
        self.q = h2 * (h2 > 0)  # ReLU activation, x2

        # print("MODEL: Q", self.q.shape, self.q)

        # Choose action based on epsilon-greedy policy
        possible_a = self.allowed_actions
        q_a = self.q[possible_a]

        if np.random.random() < self.epsilon_forward:  # explore randomly
            # print("MODEL: random action")
            sel_a = possible_a[np.random.randint(possible_a.size)]
        else:  # exploit greedily
            # print("MODEL: greedy action")
            argmax = np.argmax(q_a)
            # print("MODEL: argmax", argmax, "of", q_a, "for", possible_a)
            sel_a = possible_a[argmax]

        # TODO: originally decay every episode! n = episode number
        #  v3 - put in controller
        # self.epsilon_forward = self.epsilon_0 / (1 + self.eps_decay * step_n)  # decay epsilon
        self.epsilon_forward = self.epsilon_0
        return sel_a, self.q

    def backward(self, fingerprint, q_err):
        # Backpropagation of error through neural network
        # print("MODEL back: fp err", fingerprint.shape, q_err.shape)
        delta2 = (self.q > 0) * q_err  # derivative ReLU: 1 if x > 0 else 0
        # print("MODEL back: d2", delta2.shape)
        delta_weights2 = np.outer(self.x1, delta2.T)
        # print("MODEL back: dw2", delta_weights2.shape)

        delta1 = (self.x1 > 0) * np.dot(self.weights2, delta2)  # ReLU
        # print("MODEL back: d1", delta1.shape)
        delta_weights1 = np.outer(fingerprint, delta1)
        # print("MODEL back: dw1", delta_weights1.shape)

        self.weights1 += self.learn_rate * delta_weights1
        self.weights2 += self.learn_rate * delta_weights2
        self.bias_weights1 += self.learn_rate * delta1
        self.bias_weights2 += self.learn_rate * delta2
