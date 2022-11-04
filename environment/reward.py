from anomaly_detection import detect_anomaly


def compute_reward(fp, done, action):
    if done:
        return 1

    clf = get_classifier()
    anomalous = detect_anomaly(clf, fp)
    if anomalous:
        return -1
    else:
        return 0
