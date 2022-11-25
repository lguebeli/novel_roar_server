from environment.anomaly_detection import train_anomaly_detection, detect_anomaly


def prepare_reward_computation():
    train_anomaly_detection()


def compute_reward(fp, done):
    if done:
        return +10

    anomalous = detect_anomaly(fp)  # int [0 1]
    print("--- Detected {} FP.".format("anomalous" if anomalous else "normal"))
    if bool(anomalous):
        return -10
    else:
        return +5
