from agent.abstract_agent import AbstractAgent
from environment.state_handling import get_num_configs
from v1.agent.model import Model


class Agent1(AbstractAgent):
    def __init__(self):
        self.model = Model()
        self.next_action = 0

        self.num_actions = get_num_configs()
        self.actions = list(range(self.num_actions))

    def predict(self, fingerprint):
        next = self.next_action
        self.next_action += 1
        # we return the last valid action; next_action zero-based, num_actions one-based
        is_last = self.next_action >= self.num_actions
        return next, is_last

    def update_weights(self, reward):
        pass
