from v1.agent.agent import AgentManual
from v2.agent.agent import AgentQLearning
from v3.agent.agent import AgentAdvancedQLearning
from v4.agent.agent import AgentCorpusQLearning
from v5.agent.agent import AgentSarsa
from v99.agent.agent import AgentBruteForce
from environment.state_handling import get_prototype

AGENT = None


def get_agent():
    global AGENT
    if not AGENT:
        proto = get_prototype()
        if proto == "1":
            AGENT = AgentManual()
        elif proto == "2":
            AGENT = AgentQLearning()
        elif proto == "3":
            AGENT = AgentAdvancedQLearning()
        elif proto == "4":
            AGENT = AgentCorpusQLearning()
        elif proto == "5":
            AGENT = AgentSarsa()
        elif proto == "99":
            AGENT = AgentBruteForce()
        else:
            print("WARNING: Falling back to default agent v1!")
            AGENT = AgentManual()
    return AGENT
