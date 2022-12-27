from environment.anomaly_detection.simple_preprocessor import SimplePreprocessor
from environment.anomaly_detection.correlation_preprocessor import CorrelationPreprocessor
from environment.state_handling import get_prototype

PREPROCESSOR = None


def get_preprocessor():
    global PREPROCESSOR
    if not PREPROCESSOR:
        proto = get_prototype()
        if proto in ["1", "2", "99"]:
            PREPROCESSOR = SimplePreprocessor()
        elif proto in ["3", "4", "5"]:
            PREPROCESSOR = CorrelationPreprocessor()
        else:
            print("WARNING: Unknown prototype. Falling back to default simple preprocessor!")
            PREPROCESSOR = SimplePreprocessor()
    return PREPROCESSOR
