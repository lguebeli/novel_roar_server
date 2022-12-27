import json
import os
from abc import ABC, abstractmethod

import numpy as np

from environment.settings import CSV_FOLDER_PATH
from environment.state_handling import get_storage_path


class AbstractAgent(ABC):
    @staticmethod
    def standardize_fp(inputs):
        csv_normal = np.loadtxt(fname=os.path.join(CSV_FOLDER_PATH, "normal-behavior.csv"), delimiter=",", skiprows=1)
        norm_min = np.min(csv_normal, axis=0).reshape(-1, 1)
        norm_max = np.max(csv_normal, axis=0).reshape(-1, 1)
        # print("ABSAGENT: normal min/max", csv_normal.shape, norm_min.shape, norm_max.shape)
        # print("ABSAGENT: min/max min/max", norm_min, np.min(norm_min), np.max(norm_min), norm_max, np.min(norm_max), np.max(norm_max))

        std = (inputs - norm_min) / (norm_max - norm_min + 0.001)
        # print("ABSAGENT: input min/max", np.min(inputs), np.max(inputs))
        # print("ABSAGENT: std min/max", np.min(std), np.max(std))

        return std

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def update_weights(self, *args):
        pass


class AgentRepresentation(object):
    def __init__(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, learn_rate,
                 num_input, num_hidden, num_output):
        self.weights1 = weights1
        self.weights2 = weights2
        self.bias_weights1 = bias_weights1
        self.bias_weights2 = bias_weights2
        self.epsilon = epsilon
        self.learn_rate = learn_rate
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

    @staticmethod
    def save_agent(weights1, weights2, bias_weights1, bias_weights2, epsilon, agent, description):
        agent_file = os.path.join(get_storage_path(), "agent={}.json".format(description))
        content = {
            "weights1": weights1.tolist(),
            "weights2": weights2.tolist(),
            "bias_weights1": bias_weights1.tolist(),
            "bias_weights2": bias_weights2.tolist(),
            "epsilon": epsilon,
            "learn_rate": agent.learn_rate,
            "num_input": agent.num_input,
            "num_hidden": agent.num_hidden,
            "num_output": agent.num_output,
        }
        with open(agent_file, "w+") as file:
            json.dump(content, file)
        return agent_file
