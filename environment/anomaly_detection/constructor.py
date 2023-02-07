from environment.anomaly_detection.simple_preprocessor import SimplePreprocessor
from environment.anomaly_detection.advanced_preprocessor import AdvancedPreprocessor
from environment.state_handling import get_prototype

PREPROCESSOR = None


def get_preprocessor():
    global PREPROCESSOR
    if not PREPROCESSOR:
        proto = get_prototype()
        if proto in ["1", "2", "99"]:
            PREPROCESSOR = SimplePreprocessor()
        elif proto in ["3", "4", "5", "6", "7", "8", "98"]:
            PREPROCESSOR = AdvancedPreprocessor()
        else:
            print("WARNING: Unknown prototype. Falling back to default simple preprocessor!")
            PREPROCESSOR = SimplePreprocessor()
    return PREPROCESSOR


def reset_preprocessor():
    global PREPROCESSOR
    PREPROCESSOR = None
