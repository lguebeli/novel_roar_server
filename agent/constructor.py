from agent.agent_representation import AgentRepresentation
from agent.agent_representation_mutlilayer import AgentRepresentationMultiLayer
from environment.state_handling import get_prototype
from v1.agent.agent import AgentManual
from v2.agent.agent import AgentQLearning
from v3.agent.agent import AgentAdvancedQLearning
from v4.agent.agent import AgentCorpusQLearning
from v5.agent.agent import AgentIdealADQLearning
from v6.agent.agent import AgentSarsa
from v7.agent.agent import AgentIdealADSarsa
from v8.agent.agent import AgentOptimized

from v20.agent.agent import AgentDDQL
from v21.agent.agent import AgentDDQLIdealAD
from v22.agent.agent import AgentSarsaTabular
from v23.agent.agent import AgentIdealADSarsaTabular
from v24.agent.agent import AgentPPO
from v25.agent.agent import AgentPPOIdealAD

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
            AGENT = AgentIdealADQLearning()
        elif proto == "6":
            AGENT = AgentSarsa()
        elif proto == "7":
            AGENT = AgentIdealADSarsa()
        elif proto == "8":
            AGENT = AgentOptimized()

        elif proto == "20":
            AGENT = AgentDDQL()
        elif proto == "21":
            AGENT = AgentDDQLIdealAD()
        elif proto == "22":
            AGENT = AgentSarsaTabular()
        elif proto == "23":
            AGENT = AgentIdealADSarsaTabular()
        elif proto == "24":
            AGENT = AgentPPO()
        elif proto == "25":
            AGENT = AgentPPOIdealAD()
        else:
            print("WARNING: Unknown prototype. Falling back to default agent v1!")
            AGENT = AgentManual()
    return AGENT


def build_agent_from_repr(representation):
    proto = get_prototype()

    if proto == "21":
        assert isinstance(representation, AgentRepresentationMultiLayer), "Expected AgentRepresentationMultiLayer for proto 21"
        AGENT = AgentDDQLIdealAD(representation)
    else:
        assert isinstance(representation, AgentRepresentation), "Expected AgentRepresentation for other versions"

        if proto == "1":
            print("WARNING: Agent v1 does not support building from representation! Returning fresh agent instance...")
            AGENT = AgentManual()
        elif proto == "2":
            AGENT = AgentQLearning(representation)
        elif proto == "3":
            AGENT = AgentAdvancedQLearning(representation)
        elif proto == "4":
            AGENT = AgentCorpusQLearning(representation)
        elif proto == "5":
            AGENT = AgentIdealADQLearning(representation)
        elif proto == "6":
            AGENT = AgentSarsa(representation)
        elif proto == "7":
            AGENT = AgentIdealADSarsa(representation)
        elif proto == "8":
            AGENT = AgentOptimized(representation)
        elif proto == "20":
            AGENT = AgentDDQL(representation)
        elif proto == "22":
            AGENT = AgentSarsaTabular(representation)
        elif proto == "23":
            AGENT = AgentIdealADSarsaTabular(representation)
        elif proto == "24":
            AGENT = AgentPPO()
        elif proto == "25":
            AGENT = AgentPPOIdealAD()
        else:
            print("WARNING: Unknown prototype. Falling back to default agent v1!")
            AGENT = AgentManual()

    return AGENT