import os

import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.settings import CSV_FOLDER_PATH, RPI_MODEL_PREFIX
from environment.state_handling import get_num_configs
from v2.agent.model import ModelQLearning

EPSILON = 0.2
LEARN_RATE = 0.025  # simpl: 0.05; full: 0.025
DISCOUNT_FACTOR = 0.75  # simpl: 0.5; full: 0.75


class AgentQLearning(AbstractAgent):
    def __init__(self):
        self.model = ModelQLearning(epsilon=EPSILON, learn_rate=LEARN_RATE)

        num_configs = get_num_configs()
        self.actions = list(range(num_configs))
        self.output_size = num_configs

    def __crop_fp(self, fp):  # TODO: v3 - remove simplification
        with open(os.path.join(CSV_FOLDER_PATH, RPI_MODEL_PREFIX + "normal-behavior.csv"), "r") as csv_normal:
            csv_headers = csv_normal.read().split(",")
        headers = ["cpu_id", "tasks_running", "mem_free", "cpu_temp", "block:block_bio_remap",
                   "sched:sched_process_exec", "writeback:writeback_pages_written"]
        indexes = []
        for header in headers:
            indexes.append(csv_headers.index(header))
        return fp[indexes]

    def predict(self, fingerprint):
        std_fp = AbstractAgent.standardize_inputs(fingerprint)
        cropped_fp = self.__crop_fp(std_fp)  # TODO: v3 - remove simplification
        selected_action, q_values = self.model.forward(std_fp)
        best_action = np.argmax(q_values)

        return selected_action, best_action, q_values

    def update_weights(self, fingerprint, error):
        std_fp = AbstractAgent.standardize_inputs(fingerprint)
        cropped_fp = self.__crop_fp(std_fp)  # TODO: v3 - remove simplification
        self.model.backward(std_fp, error)

    def init_error(self):
        return np.zeros((self.output_size, 1))

    def update_error(self, error, reward, is_done, selected_action, selected_q_value, best_next_action,
                     best_next_q_value):
        # print("AGENT: R sel selval best bestval", reward, selected_action, selected_q_value, best_next_action,
        #       best_next_q_value)
        if is_done:
            error[selected_action] = reward - selected_q_value
        else:
            error[selected_action] = reward + (DISCOUNT_FACTOR * best_next_q_value) - selected_q_value  # off-policy
        # print("AGENT: err\n", error.T)
        return error
