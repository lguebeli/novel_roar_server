from pyod.models.iforest import IForest

from environment.anomaly_detection.simple_preprocessor import SimplePreprocessor
from environment.anomaly_detection.advanced_preprocessor import AdvancedPreprocessor
from environment.state_handling import get_prototype

# ========================================
# ==========   CONFIG   ==========
# ========================================
CONTAMINATION_FACTOR = 0.05

# ========================================
# ==========   GLOBALS   ==========
# ========================================
CLASSIFIER = None
PREPROCESSOR = None


def get_preprocessor():
    global PREPROCESSOR
    if not PREPROCESSOR:
        proto = get_prototype()
        if proto in ["1", "2", "99"]:
            PREPROCESSOR = SimplePreprocessor()
        elif proto in ["3", "4", "5", "6", "7", "8", "9", "98"]:
            PREPROCESSOR = AdvancedPreprocessor()
        else:
            print("WARNING: Unknown prototype. Falling back to default simple preprocessor!")
            PREPROCESSOR = SimplePreprocessor()
    return PREPROCESSOR


def reset_preprocessor():
    global PREPROCESSOR
    PREPROCESSOR = None


def get_classifier():
    global CLASSIFIER
    if not CLASSIFIER:
        proto = get_prototype()
        if proto in ["1", "2", "3", "4", "5", "6", "7", "8", "98", "99"]:
            CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
        elif proto in ["9"]:
            CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
        else:
            print("WARNING: Unknown prototype. Falling back to Isolation Forest classifier!")
            CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
    return CLASSIFIER
