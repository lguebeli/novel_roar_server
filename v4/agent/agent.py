import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.settings import ALL_CSV_HEADERS, DUPLICATE_HEADERS, DROP_CONNECTIVITY, DROP_TEMPORAL, \
    DROP_CORRELATED, DROP_ADDITIONAL
from environment.state_handling import get_num_configs
from v4.agent.model import ModelCorpusQLearning

FP_DIMS = 78
HIDDEN_NEURONS = 40

LEARN_RATE = 0.0025
DISCOUNT_FACTOR = 0.75


class AgentCorpusQLearning(AbstractAgent):
    def __init__(self):
        num_configs = get_num_configs()
        self.actions = list(range(num_configs))
        self.output_size = num_configs

        self.num_input = FP_DIMS  # Input size
        self.num_hidden = HIDDEN_NEURONS  # Hidden neurons
        self.num_output = num_configs  # Output size

        self.model = ModelCorpusQLearning(learn_rate=LEARN_RATE, num_configs=num_configs)

    def __preprocess_fp(self, fp):
        headers = ALL_CSV_HEADERS.split(",")

        duplicates = set(DUPLICATE_HEADERS)  # 3 features
        duplicates_included = []

        # 3 duplicates and 1+3+10+8=22 features dropped, leaves 103 - 25 = 78 features
        dropped_features = [*DROP_CONNECTIVITY, *DROP_TEMPORAL, *DROP_CORRELATED, *DROP_ADDITIONAL]

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
        # weights1 = np.random.uniform(0, 1, (self.num_input, self.num_hidden))
        # weights2 = np.random.uniform(0, 1, (self.num_hidden, self.num_output))

        # Xavier weight initialization
        weights1 = np.random.uniform(-1 / np.sqrt(self.num_input), +1 / np.sqrt(self.num_input),
                                     (self.num_input, self.num_hidden))
        weights2 = np.random.uniform(-1 / np.sqrt(self.num_hidden), +1 / np.sqrt(self.num_hidden),
                                     (self.num_hidden, self.num_output))

        bias_weights1 = np.zeros((self.num_hidden, 1))
        bias_weights2 = np.zeros((self.num_output, 1))

        return weights1, weights2, bias_weights1, bias_weights2

    def predict(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, state):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        hidden, q_values, selected_action = self.model.forward(weights1, weights2, bias_weights1, bias_weights2,
                                                               epsilon, inputs=ready_fp)
        return hidden, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden, weights1, weights2, bias_weights1, bias_weights2):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        new_w1, new_w2, new_bw1, new_bw2 = self.model.backward(q_values, error, hidden, weights1, weights2,
                                                               bias_weights1, bias_weights2, inputs=ready_fp)
        return new_w1, new_w2, new_bw1, new_bw2

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
