from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    @abstractmethod
    def predict(self, fingerprint):
        pass

    @abstractmethod
    def update_weights(self, fingerprint, reward):
        pass
