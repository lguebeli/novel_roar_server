import json
import os
from datetime import datetime
from time import sleep, time
import numpy as np
from tqdm import tqdm
import psutil
import threading
import csv
from utilities.plots import plot_cpu_usage, plot_memory_usage
from v23.agent.agent import AgentIdealADSarsaTabular
from environment.anomaly_detection.anomaly_detection import train_anomaly_detection
from api.configurations import map_to_ransomware_configuration, send_config
from environment.reward.ideal_AD_performance_reward import IdealADPerformanceReward
from environment.settings import MAX_EPISODES_V23, SIM_CORPUS_SIZE_V23, EPSILON_V23, DECAY_RATE_V23
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype, is_api_running, get_storage_path, get_agent_representation_path
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done
from utilities.plots import plot_average_results

DEBUG_PRINTING = False
EPSILON = EPSILON_V23
DECAY_RATE = DECAY_RATE_V23

class ControllerIdealADSarsaTabular:
    def run_c2(self):
        print("==============================\nPrepare Reward Computation\n==============================")
        train_anomaly_detection()
        if not is_simulation():
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)
        print("\n==============================\nStart Training\n==============================")
        np.random.seed(42)

        agent = AgentIdealADSarsaTabular()  # Initialize from scratch
        representation_path = get_agent_representation_path()
        if representation_path and os.path.exists(representation_path):
            agent.load_q_table(representation_path)  # Load existing Q-table if available

        self.loop_episodes(agent)
        print("\n==============================\n! Done !\n==============================")

    def loop_episodes(self, agent):
        start_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_info = "p{}-{}e-{}s".format(get_prototype(), MAX_EPISODES_V23, SIM_CORPUS_SIZE_V23)
        description = "{}={}".format(start_timestamp, run_info)
        agent_file = None

        reward_system = IdealADPerformanceReward(+10, +0, -0.2)

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []

        num_total_steps = 0
        all_start = time()

        # Resource monitoring setup
        resource_log = []
        resource_log_lock = threading.Lock()
        episode_timings = []
        stop_event = threading.Event()

        def monitor_resources():
            process = psutil.Process()  # Get current Python process
            while not stop_event.is_set():
                timestamp = time()
                try:
                    with process.oneshot():  # Optimize resource access
                        cpu_percent = process.cpu_percent(interval=0.1)  # Reduced interval to 0.1s
                        memory_info = process.memory_info()
                        memory_used_mb = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
                    with resource_log_lock:
                        resource_log.append((timestamp, cpu_percent, memory_used_mb))
                except psutil.Error:
                    # Handle cases where process info is temporarily unavailable
                    with resource_log_lock:
                        resource_log.append((timestamp, 0.0, 0.0))
                sleep(0.1)  # Ensure we don't overload the system

        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        eps_iter = range(1, MAX_EPISODES_V23 + 1) if DEBUG_PRINTING else tqdm(range(1, MAX_EPISODES_V23 + 1))
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

                if is_simulation() and sim_encryption_progress >= SIM_CORPUS_SIZE_V23:
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
            episode_timings.append((episode, eps_start, eps_end))
            log("Episode {} took: {}s, roughly {}min.".format(episode, "%.3f" % (eps_end - eps_start),
                                                              "%.1f" % ((eps_end - eps_start) / 60)))
            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)
            all_num_steps.append(steps)

            agent_file = agent.save_q_table(description=description)

        all_end = time()
        stop_event.set()
        monitor_thread.join()

        # Compute per-episode resource usage
        per_episode_resources = self.compute_per_episode_resources(episode_timings, resource_log)

        # Save resource log to CSV
        resource_log_file = os.path.join(get_storage_path(), f"resource_log_{description}.csv")
        with open(resource_log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_percent", "memory_used_mb"])
            for row in resource_log:
                writer.writerow(row)

        log("All episodes took: {}s, roughly {}min.".format("%.3f" % (all_end - all_start),
                                                            "%.1f" % ((all_end - all_start) / 60)))
        print("steps total", num_total_steps, "avg", num_total_steps / MAX_EPISODES_V23)
        print("==============================")
        print("Saving trained agent to file...")
        print("- Agent saved:", agent_file)

        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, MAX_EPISODES_V23,
                                                  description)
        print("- Plots saved:", results_plots_file)
        results_store_file = self.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps,
                                                       per_episode_resources, description)
        print("- Results saved:", results_store_file)
        print(f"- Resource log saved: {resource_log_file}")
        plot_cpu_usage(per_episode_resources, description)
        plot_memory_usage(per_episode_resources, description)
        print(f"- CPU usage plot saved: {os.path.join(get_storage_path(), f'cpu_usage_{description}.png')}")
        print(f"- Memory usage plot saved: {os.path.join(get_storage_path(), f'memory_usage_{description}.png')}")

        return agent_file, all_rewards

    @staticmethod
    def transform_fp(fp):
        """Transform fingerprint string to a 1D numpy array."""
        return np.asarray(list(map(float, fp.split(","))))

    def compute_per_episode_resources(self, episode_timings, resource_log):
        per_episode_resources = []
        last_max_memory = 400.0  # Default value based on data trend (around 400 MB)
        last_max_cpu = 50.0  # Default value for CPU usage based on data trend (around 50%-100%)

        for episode, start, end in episode_timings:
            episode_resources = [r for r in resource_log if start <= r[0] <= end]
            if episode_resources:
                timestamps, cpu_percents, memory_used_mbs = zip(*episode_resources)
                avg_cpu = np.mean(cpu_percents)
                max_cpu = np.max(cpu_percents)
                avg_memory_used_mb = np.mean(memory_used_mbs)
                max_memory_used_mb = np.max(memory_used_mbs)
                last_max_memory = max_memory_used_mb  # Update the last known memory value
                last_max_cpu = max_cpu  # Update the last known CPU value
            else:
                # If no resources are found for this episode, carry forward the last known values
                avg_cpu = last_max_cpu  # Use the last known CPU value
                max_cpu = last_max_cpu  # Use the last known CPU value
                avg_memory_used_mb = last_max_memory  # Use the last known memory value
                max_memory_used_mb = last_max_memory  # Use the last known memory value

            per_episode_resources.append({
                "episode": episode,
                "avg_cpu": float(avg_cpu),
                "max_cpu": float(max_cpu),
                "avg_memory_used_mb": float(avg_memory_used_mb),
                "max_memory_used_mb": float(max_memory_used_mb)
            })
        return per_episode_resources

    @staticmethod
    def save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps, per_episode_resources, run_description):
        results_content = json.dumps({
            "summed_rewards": all_summed_rewards,
            "avg_rewards": all_avg_rewards,
            "num_steps": all_num_steps,
            "resource_usage": per_episode_resources
        }, indent=4)
        results_file = os.path.join(get_storage_path(), f"results-store={run_description}.txt")
        with open(results_file, "w") as file:
            file.write(results_content)
        return results_file

def log(*args):
    if DEBUG_PRINTING:
        print(*args)