from environment.anomaly_detection import train_anomaly_detection, detect_anomaly


class RewardSystem(object):
    def __init__(self, r_done, r_hidden, r_detected):
        self.r_done = r_done
        self.r_hidden = r_hidden
        self.r_detected = r_detected

    @staticmethod
    def prepare_reward_computation():
        train_anomaly_detection()

    def compute_reward(self, fp, done):
        if done:
            return self.r_done

        anomalous = detect_anomaly(fp)  # int [0 1]
        print("--- Detected {} FP.".format("anomalous" if anomalous else "normal"))
        if bool(anomalous):
            return self.r_detected
        else:
            return self.r_hidden
