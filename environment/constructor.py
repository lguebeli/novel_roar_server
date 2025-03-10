from v1.environment.controller import ControllerManual
from v2.environment.controller import ControllerQLearning
from v3.environment.controller import ControllerAdvancedQLearning
from v4.environment.controller import ControllerCorpusQLearning
from v5.environment.controller import ControllerIdealADQLearning
from v6.environment.controller import ControllerSarsa
from v7.environment.controller import ControllerIdealADSarsa
from v8.environment.controller import ControllerOptimized
from v9.environment.controller import ControllerOptimizedQLearningAE
from v10.environment.controller import ControllerOptimizedQLearningIF
from v98.environment.controller import ControllerOneStepEpisodeQLearning
from v99.environment.controller import ControllerBruteForce
from v20.environment.controller import ControllerDDQL
from v21.environment.controller import ControllerDDQLwithPER
from v22.environment.controller import ControllerIdealADSarsaTabular
from v23.environment.controller import ControllerPPO
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
        elif proto == "9":
            CONTROLLER = ControllerOptimizedQLearningAE()
        elif proto == "10":
            CONTROLLER = ControllerOptimizedQLearningIF()
        elif proto == "98":
            CONTROLLER = ControllerOneStepEpisodeQLearning()
        elif proto == "99":
            CONTROLLER = ControllerBruteForce()
        elif proto == "20":
            CONTROLLER = ControllerDDQL()
        elif proto == "21":
            CONTROLLER = ControllerDDQLwithPER()
        elif proto == "22":
            CONTROLLER = ControllerIdealADSarsaTabular()
        elif proto == "23":
            CONTROLLER = ControllerPPO()
        else:
            print("WARNING: Unknown prototype. Falling back to default controller v1!")
            CONTROLLER = ControllerManual()
    return CONTROLLER
