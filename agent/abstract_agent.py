from abc import ABC, abstractmethod

import numpy as np


class AbstractAgent(ABC):
    @staticmethod
    def standardize_inputs(inputs):
        return np.log(inputs + 0.00001)  # lower value span

    @abstractmethod
    def predict(self, fingerprint):
        pass

    @abstractmethod
    def update_weights(self, fingerprint, reward):
        pass
