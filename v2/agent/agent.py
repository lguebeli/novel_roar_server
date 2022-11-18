import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.state_handling import get_num_configs
from v2.agent.model import Model

EPSILON = 0.2
LEARN_RATE = 0.01  # 0.0035
DECAY_RATE = 0.01  # 0.0005


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

        # Hyper-parameters
        self.discount = 0.85  # discount factor

    def predict(self, fingerprint):
        self.steps += 1
        self.selected_action, self.q_values = self.model.forward(fingerprint, self.steps)

        # TODO: v3 - save predicted and optimal action
        return self.selected_action

    def update_weights(self, fingerprint, error):
        self.model.backward(fingerprint, error)

    def init_error(self):
        return np.zeros(self.output_size)

    def update_error(self, error, next_action, reward, is_done):
        if is_done:
            error[self.selected_action] = reward - self.q_values[self.selected_action]
        else:
            error[self.selected_action] = \
                reward + (self.discount * np.max(next_action)) - self.q_values[self.selected_action]
        return error
