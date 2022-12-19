from time import sleep, time

from tqdm import tqdm  # add progress bar to episodes

from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward.performance_reward import PerformanceReward
from environment.settings import MAX_EPISODES_V3, MAX_STEPS_V3
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done
from utilities.plots import plot_results
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done

DEBUG_PRINTING = False

EPSILON = 0.2
DECAY_RATE = 0.01


class ControllerAdvancedQLearning(AbstractController):
    def loop_episodes(self, agent):
        reward_system = PerformanceReward(+50, +20, -1)
        weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()

        # ==============================
        # Setup collectibles
        # ==============================

        all_rewards = []
        all_summed_rewards = []
        all_num_steps = []

        last_q_values = []
        num_total_steps = 0
        all_start = time()

        eps_iter = range(1, MAX_EPISODES_V3 + 1) if DEBUG_PRINTING else tqdm(range(1, MAX_EPISODES_V3 + 1))
        for episode in eps_iter:
            # ==============================
            # Setup environment
            # ==============================

            set_rw_done(False)

            epsilon_episode = EPSILON / (1 + DECAY_RATE * episode)  # decay epsilon

            last_action = -1
            reward_store = []
            summed_reward = 0

            steps = 1
            sim_step = 1
            eps_start = time()

            # accept initial FP
            # log("Wait for initial FP...")
            if is_simulation():
                simulate_sending_fp(0)
            while not is_fp_ready():
                sleep(.5)
            curr_fp = collect_fingerprint()
            set_fp_ready(False)

            # log("Loop episode...")
            while not is_rw_done():
                # log("==================================================")
                # ==============================
                # Predict action
                # ==============================

                # transform FP into np array
                state = AbstractController.transform_fp(curr_fp)

                # agent selects action based on state
                # log("Predict action.")
                curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                            bias_weights2, epsilon_episode, state=state)
                log("Predicted action {}. Episode {} step {}.".format(selected_action, episode, sim_step))

                # ==============================
                # Take step and observe new state
                # ==============================

                # convert action to config and send to client
                if selected_action != last_action:
                    # log("Sending new action {} to client.".format(selected_action))
                    config = map_to_ransomware_configuration(selected_action)
                    if not is_simulation():  # cannot send if no socket listening during simulation
                        send_config(config)
                last_action = selected_action

                sim_step += 1
                steps += 1

                # receive next FP and compute reward based on FP
                # log("Wait for FP...")
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

                if is_simulation() and sim_step > MAX_STEPS_V3:
                    simulate_sending_rw_done()

                # log("Computing reward for next FP.")
                reward, detected = reward_system.compute_reward(next_state, is_rw_done())
                # log("Computed reward", reward)
                reward_store.append((selected_action, reward))
                summed_reward += reward
                if detected:
                    set_rw_done()  # terminate episode

                # ==============================
                # Next Q-values, error, and learning
                # ==============================

                # initialize error
                error = agent.init_error()

                if is_rw_done():
                    # update error based on observed reward
                    error = agent.update_error(error, reward, selected_action, curr_q_values,
                                               next_q_values=None, is_done=True)

                    # send error to agent, update weights accordingly
                    weights1, weights2, bias_weights1, bias_weights2 = agent.update_weights(curr_q_values, error, state,
                                                                                            curr_hidden, weights1,
                                                                                            weights2, bias_weights1,
                                                                                            bias_weights2)
                    log("Episode Q-Values:\n", curr_q_values)
                    last_q_values = curr_q_values
                else:
                    # predict next Q-values and action
                    # log("Predict next action.")
                    next_hidden, next_q_values, next_action = agent.predict(weights1, weights2, bias_weights1,
                                                                            bias_weights2, epsilon_episode,
                                                                            state=next_state)
                    # log("Predicted next action", next_action)

                    # update error based on observed reward
                    error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values,
                                               is_done=False)

                    # send error to agent, update weights accordingly
                    weights1, weights2, bias_weights1, bias_weights2 = agent.update_weights(curr_q_values, error, state,
                                                                                            curr_hidden, weights1,
                                                                                            weights2, bias_weights1,
                                                                                            bias_weights2)

                # ==============================
                # Prepare next step
                # ==============================

                # update current state
                curr_fp = next_fp
                # ========== END OF STEP ==========

            # ========== END OF EPISODE ==========
            eps_end = time()
            log("Episode {} took: {}s, roughly {}min.".format(episode, "%.3f" % (eps_end - eps_start),
                                                              "%.1f" % ((eps_end - eps_start) / 60)))
            # print("Episode {} had {} steps.".format(episode, steps))
            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_num_steps.append(steps)

        # ========== END OF TRAINING ==========
        all_end = time()
        log("All episodes took: {}s, roughly {}min.".format("%.3f" % (all_end - all_start),
                                                            "%.1f" % ((all_end - all_start) / 60)))
        print("steps total", num_total_steps, "avg", num_total_steps / MAX_EPISODES_V3)
        print("==============================\nGenerating plots...")
        # print("Rewards", all_summed_rewards)
        # print("Steps", all_num_steps)
        plot_results(all_summed_rewards, all_num_steps, MAX_EPISODES_V3, MAX_STEPS_V3)
        print("- Plots saved.")
        return last_q_values, all_rewards


def log(*args):
    if DEBUG_PRINTING:  # tqdm replaces progress inline, so prints would spam the console with multiple progress bars
        print(*args)
