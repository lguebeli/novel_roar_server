import os

from agent.model import Model

AGENT = None


class Agent(object):
    def __init__(self):
        self.model = Model()
        self.next_action = 0

        nr_of_configs = len(os.listdir(os.path.join(os.path.abspath(os.path.curdir), "rw-configs")))
        self.actions = list(range(nr_of_configs))

    def predict(self, fingerprint):
        next = self.next_action
        self.next_action += 1
        return next

    def update_weights(self, reward):
        pass


def get_agent():
    global AGENT
    if not AGENT:
        AGENT = Agent()
    return AGENT
