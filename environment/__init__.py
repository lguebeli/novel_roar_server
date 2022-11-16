from v1.environment.controller import Controller1
# from v2.environment.controller import Controller2
from environment.state_handling import get_prototype

CONTROLLER = None


def get_controller():
    global CONTROLLER
    if not CONTROLLER:
        proto = get_prototype()
        if proto == 1:
            CONTROLLER = Controller1()
        # elif proto == 2:
        #     CONTROLLER = Controller2()
        else:
            CONTROLLER = Controller1()
    return CONTROLLER
