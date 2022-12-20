from v1.environment.controller import ControllerManual
from v2.environment.controller import ControllerQLearning
from v3.environment.controller import ControllerAdvancedQLearning
from v5.environment.controller import ControllerSarsa
from v99.environment.controller import ControllerBruteForce
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
            CONTROLLER = ControllerSarsa()
        elif proto == "99":
            CONTROLLER = ControllerBruteForce()
        else:
            print("WARNING: Falling back to default controller v1!")
            CONTROLLER = ControllerManual()
    return CONTROLLER
