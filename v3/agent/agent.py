import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.settings import ALL_CSV_HEADERS, DUPLICATE_HEADERS
from environment.state_handling import get_num_configs
from v3.agent.model import ModelAdvancedQLearning

FP_DIMS = 97
HIDDEN_NEURONS = 50

EPSILON = 0.2
LEARN_RATE = 0.005
DISCOUNT_FACTOR = 0.75


class AgentAdvancedQLearning(AbstractAgent):
    def __init__(self):
        num_configs = get_num_configs()
        self.actions = list(range(num_configs))
        self.output_size = num_configs

        self.num_input = FP_DIMS  # Input size
        self.num_hidden = HIDDEN_NEURONS  # Hidden neurons
        self.num_output = num_configs  # Output size

        self.model = ModelAdvancedQLearning(epsilon=EPSILON, learn_rate=LEARN_RATE, num_configs=num_configs)

    def __preprocess_fp(self, fp):
        headers = ALL_CSV_HEADERS.split(",")

        duplicates = set(DUPLICATE_HEADERS)  # 3 features
        duplicates_included = []

        # only time metrics and duplicates
        # 3+3 features dropped, leaves 103 - 6 = 97 features
        dropped_features = ["time", "timestamp", "seconds"]

        indexes = []
        for header, value in zip(headers, fp):
            if header not in dropped_features:
                if header not in duplicates:
                    indexes.append(headers.index(header))
                else:
                    if header not in duplicates_included:
                        indexes.append(headers.index(header))
                        duplicates_included.append(header)

        return fp[indexes]

    def initialize_network(self):
        weights1 = np.random.uniform(0, 1, (self.num_input, self.num_hidden))
        weights2 = np.random.uniform(0, 1, (self.num_hidden, self.num_output))

        bias_weights1 = np.zeros((self.num_hidden, 1))
        bias_weights2 = np.zeros((self.num_output, 1))

        return weights1, weights2, bias_weights1, bias_weights2

    def predict(self, weights1, weights2, bias_weights1, bias_weights2, state):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        hidden, q_values, selected_action = self.model.forward(weights1, weights2, bias_weights1, bias_weights2, inputs=ready_fp)
        return hidden, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden, weights1, weights2, bias_weights1, bias_weights2):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        self.model.backward(q_values, error, hidden, weights1, weights2, bias_weights1, bias_weights2, inputs=ready_fp)

    def init_error(self):
        return np.zeros((self.output_size, 1))

    def update_error(self, error, reward, selected_action, curr_q_values, next_q_values, is_done):
        # print("AGENT: R sel selval best bestval", reward, selected_action, curr_q_values, next_q_values)
        if is_done:
            error[selected_action] = reward - curr_q_values[selected_action]
        else:
            # off-policy
            error[selected_action] = reward + (DISCOUNT_FACTOR * np.max(next_q_values)) - curr_q_values[selected_action]
        # print("AGENT: err\n", error.T)
        return error
