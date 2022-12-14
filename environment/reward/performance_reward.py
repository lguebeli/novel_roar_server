import math

from environment.anomaly_detection import detect_anomaly
from environment.reward.abstract_reward import AbstractReward
from environment.state_handling import collect_rate


class PerformanceReward(AbstractReward):
    def __init__(self, r_done, r_hidden, r_detected):
        self.r_done = r_done
        self.r_hidden = r_hidden
        self.r_detected = r_detected

    def compute_reward(self, fp, done):
        rate = collect_rate()
        print("REWARD: rate", rate)

        if done:
            return self.r_done

        anomalous = detect_anomaly(fp)  # int [0 1]
        print("--- Detected {} FP.".format("anomalous" if anomalous else "normal"))
        if bool(anomalous):
            print("REWARD: det", self.r_detected, max(rate, 1))
            return -(abs(self.r_detected) / max(rate, 1)) - abs(self.r_detected)  # -d/r - r
        else:
            print("REWARD: hid", rate, math.log10(rate+1), self.r_hidden)
            return math.log10(rate + 1) + abs(self.r_hidden)  # log(r+1) + h
