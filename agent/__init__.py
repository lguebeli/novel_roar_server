from v1.agent.agent import Agent1Manual
from v2.agent.agent import Agent2BruteForce
from v3.agent.agent import Agent3QLearning
from environment.state_handling import get_prototype

AGENT = None


def get_agent():
    global AGENT
    if not AGENT:
        proto = get_prototype()
        if proto == "1":
            AGENT = Agent1Manual()
        elif proto == "2":
            AGENT = Agent2BruteForce()
        elif proto == "3":
            AGENT = Agent3QLearning()
        else:
            AGENT = Agent1Manual()
    return AGENT
