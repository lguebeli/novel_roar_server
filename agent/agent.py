AGENT = None


# TODO: Agent should be the wrapper and contain a Model (pass through where necessary)
class Agent(object):
    def __init__(self):
        self.model = None
        self.selected_action = None
        self.best_action = None  # TODO: check if required, maybe only local in predict stored in one var
        pass

    def predict(self, fingerprint):
        # TODO: later on for off-policy, return both the selected and the predicted best action
        pass

    def update_weights(self, reward):
        pass


def get_agent():
    global AGENT
    if not AGENT:
        AGENT = Agent()
    return AGENT
