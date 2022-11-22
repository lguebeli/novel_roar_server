import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.state_handling import get_num_configs
from v2.agent.model import Model

EPSILON = 0.2
LEARN_RATE = 0.01  # 0.0035
DECAY_RATE = 0.01  # 0.0005
DISCOUNT_FACTOR = 0.85


class Agent2QLearning(AbstractAgent):
    def __init__(self):
        self.model = Model(epsilon=0, learn_rate=0, decay_rate=0)

        num_configs = get_num_configs()
        self.actions = list(range(num_configs))
        self.selected_action = 0
        # self.best_action = 0  # TODO: v3 - check if required, maybe only local in predict stored in one var

        self.output_size = num_configs
        self.q_values = np.zeros(num_configs)

        self.steps = 0

    def predict(self, fingerprint):
        std_fp = AbstractAgent.standardize_inputs(fingerprint)
        self.steps += 1
        self.selected_action, self.q_values = self.model.forward(std_fp, self.steps)

        # TODO: v3 - save predicted and optimal action
        return self.selected_action

    def update_weights(self, fingerprint, error):
        std_fp = AbstractAgent.standardize_inputs(fingerprint)
        self.model.backward(std_fp, error)

    def init_error(self):
        return np.zeros((self.output_size, 1))

    def update_error(self, error, next_action, reward, is_done):
        # print("AGENT q err:", self.q_values.shape, error.shape)
        if is_done:
            error[self.selected_action] = reward - self.q_values[self.selected_action]
        else:
            error[self.selected_action] = \
                reward + (DISCOUNT_FACTOR * np.max(next_action)) - self.q_values[self.selected_action]
        # print("AGENT err:", error)
        return error
