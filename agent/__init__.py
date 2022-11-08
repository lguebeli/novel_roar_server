from v1.agent.agent import Agent1
# from v2.agent.agent import Agent2
from environment.state_handling import get_prototype

AGENT = None


def get_agent():
    global AGENT
    if not AGENT:
        proto = get_prototype()
        if proto == 1:
            AGENT = Agent1()
        # elif proto == 2:
        #     AGENT = Agent2()
        else:
            AGENT = Agent1()
    return AGENT
