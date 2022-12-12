import os

import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.settings import CSV_FOLDER_PATH, RPI_MODEL_PREFIX
from environment.state_handling import get_num_configs
from v2.agent.model import ModelQLearning

EPSILON = 0.2
LEARN_RATE = 0.05  # simpl: 0.05; full: 0.025
DISCOUNT_FACTOR = 0.75  # simpl: 0.5; full: 0.75

FP_DIMS = 103  # simpl: 7; full: 103
HIDDEN_NEURONS = 60  # simpl: 10; full: 60


class AgentQLearning(AbstractAgent):
    def __init__(self):
        num_configs = get_num_configs()
        self.actions = list(range(num_configs))
        self.output_size = num_configs

        self.num_input = FP_DIMS  # Input size
        self.num_hidden = HIDDEN_NEURONS  # Hidden neurons
        self.num_output = num_configs  # Output size

        self.model = ModelQLearning(epsilon=EPSILON, learn_rate=LEARN_RATE, num_configs=num_configs)

    def __crop_fp(self, fp):  # TODO: v3 - remove simplification
        with open(os.path.join(CSV_FOLDER_PATH, RPI_MODEL_PREFIX + "normal-behavior.csv"), "r") as csv_normal:
            csv_headers = csv_normal.read().split(",")
        headers = ["cpu_id", "tasks_running", "mem_free", "cpu_temp", "block:block_bio_remap",
                   "sched:sched_process_exec", "writeback:writeback_pages_written"]
        indexes = []
        for header in headers:
            indexes.append(csv_headers.index(header))
        return fp[indexes]

    def initialize_network(self):
        weights1 = np.random.uniform(0, 1, (self.num_input, self.num_hidden))
        weights2 = np.random.uniform(0, 1, (self.num_hidden, self.num_output))

        bias_weights1 = np.zeros((self.num_hidden, 1))
        bias_weights2 = np.zeros((self.num_output, 1))

        return weights1, weights2, bias_weights1, bias_weights2

    def predict(self, weights1, weights2, bias_weights1, bias_weights2, inputs):
        std_fp = AbstractAgent.standardize_inputs(inputs)
        cropped_fp = self.__crop_fp(std_fp)  # TODO: v3 - remove simplification
        hidden, q_values, selected_action = self.model.forward(weights1, weights2, bias_weights1, bias_weights2, inputs=std_fp)
        return hidden, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden, weights1, weights2, bias_weights1, bias_weights2):
        std_fp = AbstractAgent.standardize_inputs(state)
        cropped_fp = self.__crop_fp(std_fp)  # TODO: v3 - remove simplification
        self.model.backward(q_values, error, hidden, weights1, weights2, bias_weights1, bias_weights2, inputs=std_fp)

    def init_error(self):
        return np.zeros((self.output_size, 1))

    def update_error(self, error, reward, selected_action, curr_q_values, next_q_values, is_done):
        # print("AGENT: R sel selval best bestval", reward, selected_action, curr_q_values, next_q_values)
        if is_done:
            error[selected_action] = reward - curr_q_values[selected_action]
        else:
            # off-policy
            error[selected_action] = reward + (DISCOUNT_FACTOR * np.max(next_q_values)) - curr_q_values[selected_action]
        print("AGENT: err\n", error.T)
        return error
