from datetime import datetime
from time import sleep, time
from tqdm import tqdm
from agent.agent_representation_mutlilayer import AgentRepresentationMultiLayer
from api.configurations import map_to_ransomware_configuration, send_config
from api.ransomware import send_reset_corpus, send_terminate
from environment.abstract_controller import AbstractController
from environment.reward.ideal_AD_performance_reward import IdealADPerformanceReward
from environment.settings import MAX_EPISODES_V21, SIM_CORPUS_SIZE_V21
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype
from utilities.plots import plot_average_results
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done

DEBUG_PRINTING = False
EPSILON = 0.4
DECAY_RATE = 0.01

class ControllerDDQLIdealAD(AbstractController):
    def loop_episodes(self, agent):
        start_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_info = "p{}-{}e-{}s".format(get_prototype(), MAX_EPISODES_V21, SIM_CORPUS_SIZE_V21)
        description = "{}={}".format(start_timestamp, run_info)
        agent_file = None
        simulated = is_simulation()

        reward_system = IdealADPerformanceReward(+1000, +0, -20)
        weights_list, bias_weights_list = agent.initialize_network()
        target_weights_list = [w.copy() for w in weights_list]
        target_bias_weights_list = [bw.copy() for bw in bias_weights_list]

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []
        last_q_values = []
        num_total_steps = 0
        all_start = time()

        eps_iter = range(1, MAX_EPISODES_V21 + 1) if DEBUG_PRINTING else tqdm(range(1, MAX_EPISODES_V21 + 1))
        for episode in eps_iter:
            set_rw_done(False)
            epsilon_episode = EPSILON / (1 + DECAY_RATE * (episode - 1))

            last_action = -1
            reward_store = []
            summed_reward = 0
            steps = 0
            sim_encryption_progress = 0
            eps_start = time()

            if simulated:
                simulate_sending_fp(0)
            while not is_fp_ready():
                sleep(.5)
            curr_fp = collect_fingerprint()
            set_fp_ready(False)

            while True:
                state = AbstractController.transform_fp(curr_fp)
                curr_hidden_list, curr_q_values, selected_action = agent.predict(weights_list, bias_weights_list, epsilon_episode, state=state)
                steps += 1

                if selected_action != last_action:
                    config = map_to_ransomware_configuration(selected_action)
                    if not simulated:
                        send_config(selected_action, config)
                last_action = selected_action

                if simulated:
                    simulate_sending_fp(selected_action)
                while not (is_fp_ready() or is_rw_done()):
                    sleep(.5)
                next_fp = collect_fingerprint() if not is_rw_done() else curr_fp
                next_state = AbstractController.transform_fp(next_fp)
                set_fp_ready(False)

                rate = collect_rate()
                sim_encryption_progress += rate

                if simulated and sim_encryption_progress >= SIM_CORPUS_SIZE_V21:
                    simulate_sending_rw_done()

                is_done = is_rw_done()
                reward, detected = reward_system.compute_reward(selected_action, is_done)
                reward_store.append((selected_action, reward))
                summed_reward += reward
                if detected:
                    if not is_done and not simulated:
                        send_reset_corpus()
                    set_rw_done()

                error = agent.init_error()
                if is_rw_done():
                    error = agent.update_error(error, reward, selected_action, curr_q_values, None, is_done=True)
                    weights_list, bias_weights_list = agent.update_weights(curr_q_values, error, state, curr_hidden_list, weights_list, bias_weights_list)
                    last_q_values = curr_q_values
                else:
                    next_hidden_list, next_q_values, next_action = agent.predict(target_weights_list, target_bias_weights_list, epsilon_episode, state=next_state, target=True)
                    error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values, is_done=False)
                    weights_list, bias_weights_list = agent.update_weights(curr_q_values, error, state, curr_hidden_list, weights_list, bias_weights_list)

                if is_rw_done():
                    break
                curr_fp = next_fp

            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)
            all_num_steps.append(steps)

            agent.update_target_network(weights_list, bias_weights_list)
            agent_file = AgentRepresentationMultiLayer.save_agent(weights_list, bias_weights_list, epsilon_episode, agent, description)

        all_end = time()
        print("steps total", num_total_steps, "avg", num_total_steps / MAX_EPISODES_V21)
        print("==============================")
        print("Saving trained agent to file...")
        print("- Agent saved:", agent_file)
        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, MAX_EPISODES_V21, description)
        print("- Plots saved:", results_plots_file)
        results_store_file = AbstractController.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps, description)
        print("- Results saved:", results_store_file)

        if not simulated:
            send_terminate()
        return last_q_values, all_rewards

def log(*args):
    if DEBUG_PRINTING:
        print(*args)