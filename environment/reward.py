from environment.anomaly_detection import train_anomaly_detection, detect_anomaly


def prepare_reward_computation():
    train_anomaly_detection()


def compute_reward(fp, done, action):
    if done:
        return 1

    anomalous = detect_anomaly(fp)  # int [0 1]
    print("Detected {} FP.".format("anomalous" if anomalous else "normal"))
    if bool(anomalous):
        return -1
    else:
        return 0
