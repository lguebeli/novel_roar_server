import json
import os
from datetime import datetime
from time import sleep, time
import numpy as np
from tqdm import tqdm
from v23.agent.agent import AgentPPO
from api.configurations import map_to_ransomware_configuration, send_config
from environment.reward.ideal_AD_performance_reward import IdealADPerformanceReward
from environment.settings import MAX_EPISODES_V23, SINGLE_EPISODE_LENGTH_V23
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype, is_api_running, get_storage_path, get_agent_representation_path
from utilities.plots import plot_average_results
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done

DEBUG_PRINTING = False

class ControllerPPO:
    def run_c2(self, agent=None):
        """Run the C2 server for PPO training with an optional agent."""
        try:
            print("==============================\nPrepare Reward Computation\n==============================")
            if not is_simulation():
                print("\nWaiting for API...")
                while not is_api_running():
                    sleep(1)
            print("\n==============================\nStart Training\n==============================")
            np.random.seed(42)
            training_agent = agent if agent is not None else AgentPPO()
            self.loop_episodes(training_agent)
        except Exception as e:
            print(f"Error in run_c2: {e}")
            raise

    def loop_episodes(self, agent):
        """Run the PPO training loop with the provided agent."""
        start_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_info = f"p{get_prototype()}-{MAX_EPISODES_V23}e-{SINGLE_EPISODE_LENGTH_V23}s"
        description = f"{start_timestamp}={run_info}"

        reward_system = IdealADPerformanceReward(+1000, +0, -20)
        self.agent = agent

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []
        num_total_steps = 0
        all_start = time()

        eps_iter = range(1, MAX_EPISODES_V23 + 1) if DEBUG_PRINTING else tqdm(range(1, MAX_EPISODES_V23 + 1))
        for episode in eps_iter:
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
                if is_simulation() and sim_encryption_progress >= SINGLE_EPISODE_LENGTH_V23:
                    if action == 3:  # Only action 3 ends with +1000
                        simulate_sending_rw_done()
                        reward, detected = reward_system.compute_reward(action, is_rw_done())
                    else:
                        reward = 0  # Neutral reward, no +1000
                        detected = False
                        set_rw_done(True)
                else:
                    reward, detected = reward_system.compute_reward(action, is_rw_done())

                print(f"Episode {episode}, Step {steps}, Action {action}, Reward {reward}")
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
            agent.update(states, actions, log_probs, advantages, returns)

            # Increment episode counter for entropy decay
            agent.current_episode += 1

        all_end = time()
        print(f"steps total {num_total_steps} avg {num_total_steps / MAX_EPISODES_V23}")
        print("==============================")
        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, MAX_EPISODES_V23,
                                                  description)
        print(f"- Plots saved: {results_plots_file}")
        results_store_file = self.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps, description)
        print(f"- Results saved: {results_store_file}")

        agent_repr = {
            "num_input": agent.num_input,
            "num_hidden": agent.num_hidden,
            "num_output": agent.num_output,
            "learn_rate": agent.learn_rate,
            "clip_epsilon": agent.clip_epsilon,
            "gamma": agent.gamma,
            "lambda_": agent.lambda_,
            "value_coef": agent.value_coef,
            "entropy_coef_initial": agent.entropy_coef_initial,
            "entropy_coef_decay": agent.entropy_coef_decay,
            "epochs": agent.epochs,
            "batch_size": agent.batch_size,
            "weights1": agent.weights1.tolist(),
            "weights_policy": agent.weights_policy.tolist(),
            "weights_value": agent.weights_value.tolist(),
            "fp_features": agent.fp_features,
            "mean": agent.mean.tolist(),
            "std": agent.std.tolist()
        }
        agent_file = os.path.join(get_storage_path(), f"agent={start_timestamp}.json")
        with open(agent_file, "w") as f:
            json.dump(agent_repr, f)
        print(f"- Agent representation saved: {agent_file}")

    def transform_fp(self, fp):
        """Convert fingerprint string to NumPy array."""
        return np.asarray(list(map(float, fp.split(","))))

    def compute_returns(self, rewards, dones, gamma):
        """Compute discounted returns."""
        T = len(rewards)
        returns = np.zeros(T)
        g = 0
        for t in range(T - 1, -1, -1):
            g = rewards[t] + gamma * g * (1 - dones[t])
            returns[t] = g
        return returns

    def compute_gae(self, rewards, values, dones, states, gamma, lambda_):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        last_value = 0 if dones[-1] else self.agent.forward(states[-1])[1][0, 0]
        values = np.append(values, last_value)
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
        advantages = np.zeros(T)
        a = 0
        for t in range(T - 1, -1, -1):
            a = deltas[t] + gamma * lambda_ * (1 - dones[t]) * a
            advantages[t] = a
        return advantages

    def save_results_to_file(self, all_summed_rewards, all_avg_rewards, all_num_steps, run_description):
        """Save training results to a file."""
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