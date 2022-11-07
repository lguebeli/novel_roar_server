AGENT = None


# TODO: technically more Model class than Agent class as the Agent should be the wrapper
class Agent(object):
    def __init__(self):
        pass

    def predict(self, fingerprint):
        pass

    def update_weights(self, reward):
        pass


def get_agent():
    global AGENT
    if not AGENT:
        AGENT = Agent()
    return AGENT
