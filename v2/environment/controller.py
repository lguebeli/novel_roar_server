from time import sleep

from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward import compute_reward
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint


class Controller2(AbstractController):
    def loop_episodes(self, agent):
        # accept initial FP
        print("Wait for initial FP...")
        while not is_fp_ready():
            sleep(.5)
        curr_fp = collect_fingerprint()
        set_fp_ready(False)

        last_action = 0
        reward_store = []

        print("Loop episode...")
        while not is_rw_done():
            # ==============================
            # Predict action
            # ==============================

            # transform FP into np array
            state = AbstractController.transform_fp(curr_fp)

            # agent selects action based on state
            print("Predict next action.")
            selected_action = agent.predict(state)

            # ==============================
            # Take step and observe new state
            # ==============================

            # convert action to config and send to client
            if selected_action != last_action:
                print("Sending new action {} to client.".format(selected_action))
                config = map_to_ransomware_configuration(selected_action)
                # send_config(config)
            last_action = selected_action

            # receive next FP and compute reward based on FP
            print("Wait for FP...")
            while not (is_fp_ready() or is_rw_done()):
                sleep(.5)

            if is_rw_done():
                next_fp = curr_fp
            else:
                next_fp = collect_fingerprint()

            next_state = AbstractController.transform_fp(next_fp)
            set_fp_ready(False)

            # ==============================
            # Observe reward for new state
            # ==============================

            print("Computing reward for next FP.")
            reward = compute_reward(next_state, is_rw_done(), selected_action)
            reward_store.append(reward)

            # ==============================
            # Next Q-values, error, and learning
            # ==============================

            # initialize error
            error = agent.init_error()

            if is_rw_done():
                # update error based on observed reward
                error = agent.update_error(error, None, reward, True)

                # send error to agent, update weights accordingly
                agent.update_weights(state, error)
            else:
                # predict next Q-values
                next_action = agent.predict(next_state)

                # update error based on observed reward
                error = agent.update_error(error, next_action, reward, False)

                # send error to agent, update weights accordingly
                agent.update_weights(state, error)

            # ==============================
            # Prepare next step
            # ==============================

            # update current state
            curr_fp = next_fp
