import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.state_handling import get_num_configs
from v4.agent.model import ModelSarsa

LEARN_RATE = 0.05  # 0.0035
DISCOUNT_FACTOR = 0.5  # 0.85


class AgentSarsa(AbstractAgent):
    def __init__(self):
        self.model = ModelSarsa(learn_rate=LEARN_RATE)
        self.output_size = get_num_configs()

    def predict(self, fingerprint, epsilon):
        std_fp = AbstractAgent.standardize_inputs(fingerprint)
        selected_action, q_values = self.model.forward(std_fp, epsilon)
        best_action = np.argmax(q_values)

        return selected_action, best_action, q_values

    def update_weights(self, fingerprint, error):
        std_fp = AbstractAgent.standardize_inputs(fingerprint)
        self.model.backward(std_fp, error)

    def init_error(self):
        return np.zeros((self.output_size, 1))

    def update_error(self, error, reward, is_done, selected_action, selected_q_value, next_action, next_q_value):
        print("AGENT: R sel selval best bestval", reward, selected_action, selected_q_value, next_action, next_q_value)
        if is_done:
            error[selected_action] = reward - selected_q_value
        else:
            error[selected_action] = reward + (DISCOUNT_FACTOR * next_q_value) - selected_q_value  # on-policy
        print("AGENT: err\n", error.T)
        return error
