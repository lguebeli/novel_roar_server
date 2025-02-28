import json
import os
from datetime import datetime
from time import sleep, time

import numpy as np
from tqdm import tqdm

from v22.agent.agent import AgentIdealADSarsaTabular  # Import AgentIdealADSarsaTabular class
from agent.agent_representation import AgentRepresentation
from api.configurations import map_to_ransomware_configuration, send_config
from environment.reward.ideal_AD_performance_reward import IdealADPerformanceReward
from environment.settings import MAX_EPISODES_V22, SIM_CORPUS_SIZE_V22
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype, is_api_running, get_storage_path, get_agent_representation_path
from utilities.plots import plot_average_results
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done

DEBUG_PRINTING = False
EPSILON = 0.1
DECAY_RATE = 0.01

class ControllerIdealADSarsaTabular:
    def loop_episodes(self, agent):
        start_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_info = "p{}-{}e-{}s".format(get_prototype(), MAX_EPISODES_V22, SIM_CORPUS_SIZE_V22)
        description = "{}={}".format(start_timestamp, run_info)
        agent_file = None

        reward_system = IdealADPerformanceReward(+1000, +0, -20)

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []

        num_total_steps = 0
        all_start = time()

        eps_iter = range(1, MAX_EPISODES_V22 + 1) if DEBUG_PRINTING else tqdm(range(1, MAX_EPISODES_V22 + 1))
        for episode in eps_iter:
            set_rw_done(False)
            epsilon_episode = EPSILON / (1 + DECAY_RATE * (episode - 1))

            last_action = -1
            reward_store = []
            summed_reward = 0
            steps = 1
            sim_encryption_progress = 0
            eps_start = time()

            log("Wait for initial FP...")
            if is_simulation():
                simulate_sending_fp(0)
            while not is_fp_ready():
                sleep(.5)
            curr_fp = collect_fingerprint()
            set_fp_ready(False)

            state = self.transform_fp(curr_fp)
            selected_action, q_values = agent.predict(epsilon_episode, state)
            log("Predicted action {}. Episode {} step {}.".format(selected_action, episode, steps))

            while not is_rw_done():
                if selected_action != last_action:
                    log("Sending new action {} to client.".format(selected_action))
                    config = map_to_ransomware_configuration(selected_action)
                    if not is_simulation():
                        send_config(selected_action, config)
                last_action = selected_action

                if is_simulation():
                    simulate_sending_fp(selected_action)
                while not (is_fp_ready() or is_rw_done()):
                    sleep(.5)

                if is_rw_done():
                    next_fp = curr_fp
                else:
                    next_fp = collect_fingerprint()
                next_state = self.transform_fp(next_fp)
                set_fp_ready(False)

                rate = collect_rate()
                sim_encryption_progress += rate

                if is_simulation() and sim_encryption_progress >= SIM_CORPUS_SIZE_V22:
                    simulate_sending_rw_done()

                reward, detected = reward_system.compute_reward(selected_action, is_rw_done())
                reward_store.append((selected_action, reward))
                summed_reward += reward
                if detected:
                    set_rw_done()

                if is_rw_done():
                    next_action = None
                    next_q_value = None
                else:
                    next_action, next_q_values = agent.predict(epsilon_episode, next_state)
                    next_q_value = next_q_values[next_action]
                    steps += 1

                agent.update_q_table(state, selected_action, reward, next_state, next_action, is_rw_done())

                curr_fp = next_fp
                selected_action = next_action
                state = next_state

            eps_end = time()
            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)
            all_num_steps.append(steps)

            # Now use the agent instance to call save_q_table
            agent_file = agent.save_q_table(description=description)

        all_end = time()
        print("steps total", num_total_steps, "avg", num_total_steps / MAX_EPISODES_V22)
        print("==============================")
        print("Saving trained agent to file...")
        print("- Agent saved:", agent_file)

        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, MAX_EPISODES_V22,
                                                  description)
        print("- Plots saved:", results_plots_file)
        results_store_file = self.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps,
                                                       description)
        print("- Results saved:", results_store_file)
        return None, all_rewards

    def run_c2(self):
        print("==============================\nPrepare Reward Computation\n==============================")
        if not is_simulation():
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)
        print("\n==============================\nStart Training\n==============================")
        np.random.seed(42)

        representation_path = get_agent_representation_path()
        if representation_path and os.path.exists(representation_path):
            with open(representation_path, "r") as agent_file:
                repr_dict = json.load(agent_file)
            representation = AgentRepresentation(repr_dict["weights1"], repr_dict["weights2"],
                                                 repr_dict["bias_weights1"], repr_dict["bias_weights2"],
                                                 repr_dict["epsilon"], repr_dict["learn_rate"],
                                                 repr_dict["num_input"], repr_dict["num_hidden"],
                                                 repr_dict["num_output"])
            agent = AgentRepresentation.build_agent_from_repr(representation)
        else:
            # Create agent from scratch if no pre-trained model exists
            agent = AgentIdealADSarsaTabular()  # Initialize AgentIdealADSarsaTabular

        self.loop_episodes(agent)
        print("\n==============================\n! Done !\n==============================")

    @staticmethod
    def transform_fp(fp):
        return np.asarray(list(map(float, fp.split(",")))).reshape(-1, 1)

    @staticmethod
    def save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps, run_description):
        results_content = json.dumps({
            "summed_rewards": all_summed_rewards,
            "avg_rewards": all_avg_rewards,
            "num_steps": all_num_steps
        }, indent=4)
        results_file = os.path.join(get_storage_path(), "results-store={}.txt".format(run_description))
        with open(results_file, "w") as file:
            file.write(results_content)
        return results_file

def log(*args):
    if DEBUG_PRINTING:
        print(*args)
