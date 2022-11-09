from time import sleep

import numpy as np

from agent.abstract_agent import AbstractAgent
from api.configurations import map_to_ransomware_configuration, send_config
from environment.reward import compute_reward
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, get_num_configs, collect_fingerprint
from v1.agent.model import Model


class Agent1(AbstractAgent):
    def __init__(self):
        self.model = Model()
        self.next_action = 0

        self.num_actions = get_num_configs()
        self.actions = list(range(self.num_actions))

    def predict(self, fingerprint):
        next = self.next_action
        self.next_action += 1
        is_last = self.next_action > self.num_actions  # we return the last valid action
        return next, is_last

    def update_weights(self, reward):
        pass

    def loop_episodes(self):
        # accept initial FP
        # print("Wait for initial FP...")
        while not is_fp_ready():
            sleep(.5)
        curr_fp = collect_fingerprint()
        set_fp_ready(False)

        last_action = None

        # print("Loop episode...")
        while True:
            # transform FP into np array
            state = transform_fp(curr_fp)

            # agent selects action based on state
            # print("Predict next action.")
            selected_action, is_last = self.predict(state)

            # convert action to config and send to client
            if selected_action != last_action:
                # print("Sending new action {} to client.".format(selected_action))
                config = map_to_ransomware_configuration(selected_action)
                send_config(config)
            last_action = selected_action
            # TODO: store action in reward module?

            # receive next FP and compute reward based on FP
            # print("Wait for FP...")
            while not (is_fp_ready()):
                sleep(.5)

            next_fp = collect_fingerprint()
            set_fp_ready(False)

            # print("Computing reward for next FP.")
            # TODO: including action required?
            reward = compute_reward(transform_fp(next_fp), is_rw_done(), selected_action)

            # send reward to agent, update weights accordingly
            self.update_weights(reward)

            if is_last:
                # terminate episode instantly
                break
            # set next_fp to curr_fp for next iteration
            curr_fp = next_fp


def transform_fp(fp):
    split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
    return np.asarray(split_to_floats)
