import numpy as np

from environment.state_handling import get_num_configs

FP_DIMS = 103
HIDDEN_NEURONS = 200
NUM_CONFIGS = get_num_configs()


class Model(object):
    def __init__(self, learn_rate):
        # Initialize random seed for reproducibility
        np.random.seed(42)

        # Initialize hyperparams
        self.learn_rate = learn_rate

        # Initialize neural network
        self.allowed_actions = np.arange(NUM_CONFIGS)

        num_input = FP_DIMS  # Input size
        num_hidden = HIDDEN_NEURONS  # Hidden neurons
        num_output = NUM_CONFIGS  # Output size

        # Xavier weight initialization
        self.weights1 = np.random.uniform(-1/np.sqrt(num_input), 1/np.sqrt(num_input), (num_hidden+1, num_input+1))
        self.weights2 = np.random.uniform(-1/np.sqrt(num_hidden), 1/np.sqrt(num_hidden), (num_output, num_hidden+1))
        # When we have a clever initialization of W1 then we don't need to set H0=1.
        # Instead, we can instantiate weights s.t. H0=1 is guaranteed.
        # Since x0 = 1, we can set w1[0,0]=inf and w1[0,:]=0,
        # resulting in a0=inf and sig(a0)=1 (which also works with tanh).
        # Also, updating weights does not update W1[:,0]
        self.weights1[0, :] = 0

        # self.bias_weights1 = np.zeros((num_hidden, 1))
        # self.bias_weights2 = np.zeros((num_output, 1))

        self.h = np.ones((num_hidden + 1, 1))
        self.q = np.ones((num_output, 1))

    def forward(self, inputs, epsilon):
        # print("MODEL: inputs", inputs.shape, np.min(inputs), np.argmin(inputs), np.max(inputs), np.argmax(inputs), inputs)

        # ==============================
        # Q-VALUES
        # ==============================

        # Add bias neuron
        inputs = np.insert(inputs, obj=0, values=0, axis=0)

        # Forward pass through neural network to compute Q-values
        # print("MODEL: w1", self.weights1.shape, self.weights1)
        # a1 = np.dot(self.weights1.T, inputs) + self.bias_weights1
        a1 = np.dot(self.weights1, inputs)
        # print("MODEL: a1 dot", np.dot(self.weights1.T, inputs))
        # print("MODEL: a1 min/max", a1.shape, np.min(a1), np.argmin(a1), np.max(a1), np.argmax(a1))
        # self.h = a1 * (a1 > 0)  # ReLU activation, x if x > 0 else 0
        self.h = 1 / (1 + np.exp(-a1))  # logistic activation

        # print("MODEL: h min/max", self.h.shape, np.min(self.h), np.argmin(self.h), np.max(self.h), np.argmax(self.h))
        # print("MODEL: w2", self.weights2.shape, self.weights2)

        # a2 = np.dot(self.weights2.T, self.h) + self.bias_weights2
        a2 = np.dot(self.weights2, self.h)
        # print("MODEL: a2 dot", np.dot(self.weights2.T, self.h))
        # print("MODEL: a2 min/max", a2.shape, np.min(a2), np.argmin(a2), np.max(a2), np.argmax(a2))
        # self.q = a2 * (a2 > 0)  # ReLU activation
        self.q = 1 / (1 + np.exp(-a2))  # logistic activation

        print("MODEL: Q", self.q.shape, "\n", self.q)

        # ==============================
        # POLICY
        # ==============================

        # Choose action based on epsilon-greedy policy
        possible_a = self.allowed_actions  # technically an array of indexes
        print("MODEL: q and a", self.q, possible_a)
        # q_a = self.q[possible_a]
        q_a = self.q[possible_a]

        if np.random.random() < epsilon:  # explore randomly
            print("MODEL: random action")
            sel_a = possible_a[np.random.randint(possible_a.size)]
        else:  # exploit greedily
            print("MODEL: greedy action")
            argmax = np.argmax(q_a)
            # print("MODEL: argmax", argmax, "of", q_a, "for", possible_a)
            sel_a = possible_a[argmax]

        return sel_a, self.q.reshape(-1,)

    def backward(self, fingerprint, q_err):
        # Backpropagation of error through neural network

        # ==============================
        # COMPUTE DELTA
        # ==============================

        print("MODEL: back", fingerprint.shape, q_err.shape)

        # delta2 = (self.q > 0) * q_err  # derivative ReLU: 1 if x > 0 else 0
        # delta2 = self.q * (1 - self.q) * q_err  # derivative logistic: f(x) * (1 - f(x))
        # delta_weights2 = np.outer(self.h, delta2.T)
        # print("MODEL back: fp err", fingerprint.shape, q_err.shape, fingerprint, q_err)
        # print("MODEL back: d2", delta2.shape, delta2)
        # print("MODEL back: dw2", delta_weights2.shape, delta_weights2)

        # delta1 = (self.h > 0) * np.dot(self.weights2, delta2)  # ReLU
        # delta1 = self.h * (1 - self.h) * np.dot(self.weights2, delta2)  # logistic
        # delta_weights1 = np.outer(fingerprint, delta1)
        # print("MODEL back: d1", delta1.shape, delta1)
        # print("MODEL back: dw1", delta_weights1.shape, delta_weights1)

        # derivative logistic: f(x) * (1 - f(x))
        delta1 = np.dot((q_err * self.weights2 * self.h * (1 - self.h))[1, :], fingerprint)
        delta2 = q_err * self.q * (1 - self.q)

        # ==============================
        # UPDATE WEIGHTS
        # ==============================

        # print("MODEL: weights1 before", self.weights1.shape, np.min(self.weights1), np.argmin(self.weights1), np.max(self.weights1), np.argmax(self.weights1))
        # print("MODEL: weights2 before", self.weights2.shape, np.min(self.weights2), np.argmin(self.weights2), np.max(self.weights2), np.argmax(self.weights2))

        # self.weights1 += self.learn_rate * delta_weights1
        # self.weights2 += self.learn_rate * delta_weights2
        # self.bias_weights1 += self.learn_rate * delta1
        # self.bias_weights2 += self.learn_rate * delta2
        self.weights1 -= self.learn_rate * delta1
        self.weights2 -= self.learn_rate * delta2

        # print("MODEL: weights1 after", self.weights1.shape, np.min(self.weights1), np.argmin(self.weights1), np.max(self.weights1), np.argmax(self.weights1))
        # print("MODEL: weights2 after", self.weights2.shape, np.min(self.weights2), np.argmin(self.weights2), np.max(self.weights2), np.argmax(self.weights2))
