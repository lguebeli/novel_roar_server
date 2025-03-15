from v1.environment.controller import ControllerManual
from v2.environment.controller import ControllerQLearning
from v3.environment.controller import ControllerAdvancedQLearning
from v4.environment.controller import ControllerCorpusQLearning
from v5.environment.controller import ControllerIdealADQLearning
from v6.environment.controller import ControllerSarsa
from v7.environment.controller import ControllerIdealADSarsa
from v8.environment.controller import ControllerOptimized

from v20.environment.controller import ControllerDDQL
from v21.environment.controller import ControllerDDQLIdealAD
from v22.environment.controller import ControllerSarsaTabular
from v23.environment.controller import ControllerIdealADSarsaTabular
from v24.environment.controller import ControllerPPO
from v25.environment.controller import ControllerPPOIdealAD
from environment.state_handling import get_prototype

CONTROLLER = None


def get_controller():
    global CONTROLLER
    if not CONTROLLER:
        proto = get_prototype()
        if proto == "1":
            CONTROLLER = ControllerManual()
        elif proto == "2":
            CONTROLLER = ControllerQLearning()
        elif proto == "3":
            CONTROLLER = ControllerAdvancedQLearning()
        elif proto == "4":
            CONTROLLER = ControllerCorpusQLearning()
        elif proto == "5":
            CONTROLLER = ControllerIdealADQLearning()
        elif proto == "6":
            CONTROLLER = ControllerSarsa()
        elif proto == "7":
            CONTROLLER = ControllerIdealADSarsa()
        elif proto == "8":
            CONTROLLER = ControllerOptimized()

        elif proto == "20":
            CONTROLLER = ControllerDDQL()
        elif proto == "21":
            CONTROLLER = ControllerDDQLIdealAD()
        elif proto == "22":
            CONTROLLER = ControllerSarsaTabular()
        elif proto == "23":
            CONTROLLER = ControllerIdealADSarsaTabular()
        elif proto == "24":
            CONTROLLER = ControllerPPO()
        elif proto == "25":
            CONTROLLER = ControllerPPOIdealAD()
        else:
            print("WARNING: Unknown prototype. Falling back to default controller v1!")
            CONTROLLER = ControllerManual()
    return CONTROLLER
