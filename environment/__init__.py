from v1.environment.controller import Controller1
from v2.environment.controller import Controller2
from v3.environment.controller import Controller3
from v5.environment.controller import Controller5
from environment.state_handling import get_prototype

CONTROLLER = None


def get_controller():
    global CONTROLLER
    if not CONTROLLER:
        proto = get_prototype()
        if proto == "1":
            CONTROLLER = Controller1()
        elif proto == "2":
            CONTROLLER = Controller2()
        elif proto == "3":
            CONTROLLER = Controller3()
        elif proto == "5":
            CONTROLLER = Controller5()
        else:
            print("WARNING: Falling back to default controller v1!")
            CONTROLLER = Controller1()
    return CONTROLLER
