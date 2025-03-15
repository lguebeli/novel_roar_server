import numpy as np

class ModelOptimized(object):
    def __init__(self, learn_rate, num_configs, hidden_sizes):
        self.learn_rate = learn_rate
        self.allowed_actions = np.asarray(range(num_configs))
        self.hidden_sizes = hidden_sizes  # List of hidden layer sizes, e.g., [64, 32]
        self.num_layers = len(hidden_sizes) + 1  # Hidden layers + output layer

    def forward(self, weights_list, bias_weights_list, epsilon, inputs):
        # Forward pass through multiple layers
        activations = [inputs]  # Store activations for each layer
        for i in range(self.num_layers):
            pre_activation = np.dot(weights_list[i].T, activations[-1]) + bias_weights_list[i]
            if i < self.num_layers - 1:  # Hidden layers use Logistic
                activation = 1 / (1 + np.exp(-pre_activation))
            else:  # Output layer uses SiLU
                activation = pre_activation / (1 + np.exp(-pre_activation))
            activations.append(activation)

        q_values = activations[-1]  # Q-values from output layer
        hidden_list = activations[1:-1]  # Hidden layer outputs

        # Epsilon-greedy policy
        q_a = q_values[self.allowed_actions]
        if np.random.random() < epsilon:
            selected_action = self.allowed_actions[np.random.randint(self.allowed_actions.size)]
        else:
            selected_action = self.allowed_actions[np.argmax(q_a)]

        return hidden_list, q_values, selected_action

    def backward(self, q_values, q_err, hidden_list, weights_list, bias_weights_list, inputs):
        # Backpropagation through multiple layers
        activations = [inputs] + hidden_list + [q_values]
        deltas = [None] * self.num_layers

        # Output layer delta (SiLU derivative)
        sig_q = 1 / (1 + np.exp(-q_values))
        deltas[-1] = sig_q * (1 + q_values * (1 - sig_q)) * q_err

        # Hidden layer deltas (Logistic derivative)
        for i in range(self.num_layers - 2, -1, -1):
            hidden = activations[i + 1]
            deltas[i] = hidden * (1 - hidden) * np.dot(weights_list[i + 1], deltas[i + 1])

        # Update weights and biases
        new_weights_list = []
        new_bias_weights_list = []
        for i in range(self.num_layers):
            delta_weights = np.outer(activations[i], deltas[i].T)
            new_weights_list.append(weights_list[i] + self.learn_rate * delta_weights)
            new_bias_weights_list.append(bias_weights_list[i] + self.learn_rate * deltas[i])

        return new_weights_list, new_bias_weights_list