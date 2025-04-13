import json
import os
from datetime import datetime
from time import sleep, time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from v24.agent.agent import AgentPPONormalAD
from api.configurations import map_to_ransomware_configuration, send_config
from environment.reward.performance_reward import PerformanceReward
from environment.settings import MAX_EPISODES_V24, SINGLE_EPISODE_LENGTH_V24
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype, is_api_running, get_storage_path
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done
from utilities.plots import plot_average_results
from environment.evaluation.evaluation_ppo import evaluate_agent  # Import for early stopping

DEBUG_PRINTING = False

class ControllerPPONormalAD:
    def run_c2(self, agent=None, timestamp=None, patience=5, eval_freq=1000):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        print("==============================\nPrepare Reward Computation\n==============================")
        if not is_simulation():
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)
        print("\n==============================\nStart Training\n==============================")
        tf.random.set_seed(42)
        np.random.seed(42)
        training_agent = agent if agent is not None else AgentPPONormalAD()
        self.loop_episodes(training_agent, timestamp, patience, eval_freq)

    def loop_episodes(self, agent, timestamp, patience, eval_freq):
        start_timestamp = timestamp
        run_info = f"p{get_prototype()}-{MAX_EPISODES_V24}e-{SINGLE_EPISODE_LENGTH_V24}s"
        description = f"{start_timestamp}={run_info}"

        reward_system = PerformanceReward(+10, +0, -0.2)
        self.agent = agent

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []
        num_total_steps = 0
        all_start = time()

        best_accuracy = -1
        patience_counter = 0
        KNOWN_BEST_ACTION = 3

        for episode in tqdm(range(1, MAX_EPISODES_V24 + 1)):
            set_rw_done(False)
            last_action = -1
            reward_store = []
            summed_reward = 0
            steps = 0
            sim_encryption_progress = 0

            log("Wait for initial FP...")
            if is_simulation():
                simulate_sending_fp(0)
            while not is_fp_ready():
                sleep(.5)
            curr_fp = collect_fingerprint()
            set_fp_ready(False)

            state = self.transform_fp(curr_fp)
            states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

            while not is_rw_done():
                action, log_prob, value = agent.act(state)
                steps += 1
                log(f"Selected action {action}. Episode {episode} step {steps}.")

                if action != last_action:
                    config = map_to_ransomware_configuration(action)
                    if not is_simulation():
                        send_config(action, config)
                    last_action = action

                if is_simulation():
                    simulate_sending_fp(action)
                while not (is_fp_ready() or is_rw_done()):
                    sleep(0.5)
                next_fp = collect_fingerprint() if is_fp_ready() else state
                next_state = self.transform_fp(next_fp)
                set_fp_ready(False)

                rate = collect_rate()
                sim_encryption_progress += rate

                if is_simulation() and sim_encryption_progress >= SINGLE_EPISODE_LENGTH_V24:
                    simulate_sending_rw_done()
                    reward, detected = reward_system.compute_reward(next_state, is_rw_done())
                else:
                    reward, detected = reward_system.compute_reward(next_state, is_rw_done())

                reward_store.append((action, reward))
                summed_reward += reward
                if detected:
                    set_rw_done(True)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(is_rw_done())
                state = next_state

            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)
            all_num_steps.append(steps)

            states = np.array(states)
            actions = np.array(actions)
            log_probs = np.array(log_probs)
            values = np.array(values)
            rewards = np.array(rewards)
            dones = np.array(dones)

            returns = self.compute_returns(rewards, dones, agent.gamma)
            advantages = self.compute_gae(rewards, values, dones, states, agent.gamma, agent.lambda_)
            agent.update(states, actions, log_probs, values, advantages, returns)

            agent.current_episode += 1

            # Early stopping logic
            if episode % eval_freq == 0:
                accuracies_overall, _ = evaluate_agent(agent)
                current_accuracy = accuracies_overall.get(KNOWN_BEST_ACTION, 0) / accuracies_overall["total"]
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    # Save the best agent
                    with open(os.path.join(get_storage_path(), f"agent={timestamp}-best.json"), "w") as f:
                        json.dump(agent.get_weights_dict(), f)
                    patience_counter = 0
                    print(f"Episode {episode}: New best accuracy {current_accuracy:.4f}")
                else:
                    patience_counter += 1
                    print(f"Episode {episode}: Accuracy {current_accuracy:.4f}, Patience {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at episode {episode}")
                        break

        all_end = time()
        log("All episodes took: {}s, roughly {}min.".format("%.3f" % (all_end - all_start),
                                                            "%.1f" % ((all_end - all_start) / 60)))
        print(f"steps total {num_total_steps} avg {num_total_steps / MAX_EPISODES_V24}")
        print("==============================")
        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, episode,
                                                  description)
        print(f"- Plots saved: {results_plots_file}")
        results_store_file = self.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps, description)
        print(f"- Results saved: {results_store_file}")

        # Save the final agent
        agent_file = os.path.join(get_storage_path(), f"agent={timestamp}.json")
        with open(agent_file, "w") as f:
            json.dump(agent.get_weights_dict(), f)
        print(f"- Agent representation saved: {agent_file}")

    def transform_fp(self, fp):
        return np.array(list(map(float, fp.split(","))))

    def compute_returns(self, rewards, dones, gamma):
        T = len(rewards)
        returns = np.zeros(T)
        g = 0
        for t in range(T - 1, -1, -1):
            g = rewards[t] + gamma * g * (1 - dones[t])
            returns[t] = g
        return returns

    def compute_gae(self, rewards, values, dones, states, gamma, lambda_):
        T = len(rewards)
        last_value = 0 if dones[-1] else self.agent.call(tf.convert_to_tensor(states[-1], dtype=tf.float32))[1][0, 0]
        values = np.append(values, last_value)
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
        advantages = np.zeros(T)
        a = 0
        for t in range(T - 1, -1, -1):
            a = deltas[t] + gamma * lambda_ * (1 - dones[t]) * a
            advantages[t] = a

        final_advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  # Already normalized
        return final_advantages

    def save_results_to_file(self, all_summed_rewards, all_avg_rewards, all_num_steps, run_description):
        results_content = json.dumps({
            "summed_rewards": all_summed_rewards,
            "avg_rewards": all_avg_rewards,
            "num_steps": all_num_steps
        }, indent=4)
        results_file = os.path.join(get_storage_path(), f"results-store={run_description}.txt")
        with open(results_file, "w") as file:
            file.write(results_content)
        return results_file

def log(*args):
    if DEBUG_PRINTING:
        print(*args)

if __name__ == "__main__":
    controller = ControllerPPONormalAD()
    controller.run_c2()