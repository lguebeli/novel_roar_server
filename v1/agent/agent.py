import os
from time import sleep

import numpy as np

from agent.abstract_agent import AbstractAgent
from api.configurations import map_to_ransomware_configuration, send_config
from environment.reward import compute_reward
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint
from v1.agent.model import Model


class Agent1(AbstractAgent):
    def __init__(self):
        self.model = Model()
        self.next_action = 0

        nr_of_configs = len(os.listdir(os.path.join(os.path.abspath(os.path.curdir), "rw-configs")))
        self.actions = list(range(nr_of_configs))

    def predict(self, fingerprint):
        next = self.next_action
        self.next_action += 1
        return next

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
            selected_action = self.predict(state)

            # convert action to config and send to client
            if selected_action != last_action:
                # print("Sending new action {} to client.".format(selected_action))
                config = map_to_ransomware_configuration(selected_action)
                send_config(config)
            last_action = selected_action
            # TODO: store action in reward module?

            # receive next FP and compute reward based on FP
            # print("Wait for FP...")
            while not (is_fp_ready() or is_rw_done()):
                sleep(.5)

            if is_rw_done():
                # print("Computing reward for current FP.")
                # TODO: including action required?
                reward = compute_reward(transform_fp(curr_fp), is_rw_done(), selected_action)
                set_fp_ready(False)
            else:
                next_fp = collect_fingerprint()
                set_fp_ready(False)

                # print("Computing reward for next FP.")
                # TODO: including action required?
                reward = compute_reward(transform_fp(next_fp), is_rw_done(), selected_action)

            # send reward to agent, update weights accordingly
            self.update_weights(reward)

            if not is_rw_done():
                # set next_fp to curr_fp for next iteration
                curr_fp = next_fp
            else:
                # terminate episode instantly
                break


def transform_fp(fp):
    split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
    return np.asarray(split_to_floats)
