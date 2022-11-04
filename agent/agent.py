AGENT = None


class Agent(object):
    def __init__(self):
        pass

    def predict(self, fingerprint):
        pass

    def update_weights(self):
        pass


def get_agent():
    global AGENT
    if not AGENT:
        AGENT = Agent()
    return AGENT
