from time import sleep, time

from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward import RewardSystem
from environment.settings import MAX_STEPS_V2
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation
from simulate import simulate_sending_fp, simulate_sending_rw_done

USE_SIMPLE_FP = False


class ControllerQLearning(AbstractController):
    def loop_episodes(self, agent):
        # ==============================
        # Setup agent
        # ==============================

        weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()

        # ==============================
        # Setup environment
        # ==============================

        reward_system = RewardSystem(+10, +5, -10) if USE_SIMPLE_FP else RewardSystem(+50, +20, -20)
        last_action = -1
        reward_store = []

        sim_step = 1
        sim_start = time()

        # accept initial FP
        # print("Wait for initial FP...")
        if is_simulation():
            simulate_sending_fp(0)
        while not is_fp_ready():
            sleep(.5)
        curr_fp = collect_fingerprint()
        set_fp_ready(False)

        # print("Loop episode...")
        while not is_rw_done():
            print("==================================================")
            # ==============================
            # Predict action
            # ==============================

            # transform FP into np array
            state = AbstractController.transform_fp(curr_fp)

            # agent selects action based on state
            # print("Predict action.")
            curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1, bias_weights2, state=state)
            print("Predicted action {}. Step {}.".format(selected_action, sim_step))

            # ==============================
            # Take step and observe new state
            # ==============================

            # convert action to config and send to client
            if selected_action != last_action:
                # print("Sending new action {} to client.".format(selected_action))
                config = map_to_ransomware_configuration(selected_action)
                if not is_simulation():  # cannot send if no socket listening during simulation
                    send_config(config)
            last_action = selected_action

            sim_step += 1

            # receive next FP and compute reward based on FP
            # print("Wait for FP...")
            if is_simulation():
                simulate_sending_fp(selected_action)
            while not (is_fp_ready() or is_rw_done()):
                sleep(.5)

            if is_rw_done():
                next_fp = curr_fp
            else:
                next_fp = collect_fingerprint()

            # transform FP into np array
            next_state = AbstractController.transform_fp(next_fp)
            set_fp_ready(False)

            # ==============================
            # Observe reward for new state
            # ==============================

            if is_simulation() and sim_step > MAX_STEPS_V2:
                simulate_sending_rw_done()

            # print("Computing reward for next FP.")
            reward = reward_system.compute_reward(next_state, is_rw_done())
            reward_store.append((selected_action, reward))

            # ==============================
            # Next Q-values, error, and learning
            # ==============================

            # initialize error
            error = agent.init_error()

            if is_rw_done():
                # update error based on observed reward
                error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values=None, is_done=True)

                # send error to agent, update weights accordingly
                agent.update_weights(curr_q_values, error, state, curr_hidden, weights1, weights2, bias_weights1, bias_weights2)
                print("Final Q-Values:\n", curr_q_values)
            else:
                # predict next Q-values and action
                # print("Predict next action.")
                next_hidden, next_q_values, next_action = agent.predict(weights1, weights2, bias_weights1, bias_weights2, state=next_state)
                # print("Predicted next action", next_action)

                # update error based on observed reward
                error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values, is_done=False)

                # send error to agent, update weights accordingly
                agent.update_weights(curr_q_values, error, state, curr_hidden, weights1, weights2, bias_weights1, bias_weights2)

            # ==============================
            # Prepare next step
            # ==============================

            # update current state
            curr_fp = next_fp

        sim_end = time()
        print("Took: {}s, roughly {}min.".format("%.3f" % (sim_end - sim_start), "%.1f" % ((sim_end - sim_start) / 60)))
        return reward_store
