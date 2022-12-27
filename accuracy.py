import json
import os
from contextlib import redirect_stdout
from datetime import datetime
from time import time

import numpy as np
from tqdm import tqdm

from agent.abstract_agent import AgentRepresentation
from environment.constructor import get_controller
from environment.reward.abstract_reward import AbstractReward
from environment.reward.performance_reward import PerformanceReward
from environment.settings import CSV_FOLDER_PATH
from environment.state_handling import initialize_storage, cleanup_storage, set_prototype, get_num_configs, \
    collect_fingerprint, get_storage_path, set_simulation
from utilities.simulate import simulate_sending_fp
from v4.agent.agent import AgentCorpusQLearning


def evaluate_agent(agent, reward_system, weights1, weights2, bias_weights1, bias_weights2, EPSILON):
    accuracies = []
    num_configs = get_num_configs()
    for config in range(num_configs):
        print("{}Config".format("\n" if config > 0 else ""), config)
        config_fp_dir = os.path.join(CSV_FOLDER_PATH, "infected-c{}".format(config))
        fp_files = os.listdir(config_fp_dir)
        num_selected_hidden_action = 0
        for fp_file in tqdm(fp_files):
            # collect selected initial fingerprint
            with open(os.path.join(config_fp_dir, fp_file)) as file:
                fp = file.readline()[1:-1].replace(" ", "")
            state = transform_fp(fp)

            # predict next action
            curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                        bias_weights2, EPSILON, state)

            # collect next state
            simulate_sending_fp(selected_action)
            next_fp = collect_fingerprint()
            next_state = transform_fp(next_fp)

            # evaluate chosen action based on new state
            _, detected = reward_system.compute_reward(next_state, False)
            num_selected_hidden_action += int(not detected)
        accuracies.append((config, num_selected_hidden_action, len(fp_files)))
    return accuracies


def find_agent_file(timestamp):
    storage_path = get_storage_path()
    files = os.listdir(storage_path)
    filtered = list(filter(lambda f: f.startswith("agent={}".format(timestamp)), files))
    return os.path.join(storage_path, filtered.pop())


def transform_fp(fp):
    split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
    return np.asarray(split_to_floats).reshape(-1, 1)  # shape (F, 1)


def print_accuracy_table(accuracies, logs):
    for accuracy in accuracies:
        print("Config {}:".format(accuracy[0]), "{}% ({}/{})".format(
            "%.3f" % (accuracy[1] / accuracy[2] * 100), accuracy[1], accuracy[2]))
        logs.append("Config {}: {}% ({}/{})".format(
            accuracy[0], "%.3f" % (accuracy[1] / accuracy[2] * 100), accuracy[1], accuracy[2]))
    return logs


# ==============================
# SETUP
# ==============================
log_file = os.path.join(os.path.curdir, "storage",
                        "accuracy-report={}.txt".format(datetime.now().strftime("%Y-%m-%d--%H-%M-%S")))
logs = []

print("========== PREPARE ENVIRONMENT ==========\nAD evaluation is written to log file directly")

EPSILON = 0.1

initialize_storage()
try:
    set_prototype("4")
    set_simulation(True)

    # ==============================
    # WRITE AD EVALUATION TO LOG FILE
    # ==============================

    with open(log_file, "w+") as f:
        with redirect_stdout(f):
            print("========== PREPARE ENVIRONMENT ==========")
            AbstractReward.prepare_reward_computation()

    # ==============================
    # EVAL UNTRAINED AGENT
    # ==============================

    print("\n========== MEASURE ACCURACY (INITIAL) ==========")
    logs.append("\n========== MEASURE ACCURACY (INITIAL) ==========")

    agent = AgentCorpusQLearning()
    reward_system = PerformanceReward(+100, +20, -20)
    weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()
    logs.append("Agent representation")
    logs.append("> weights1: {}".format(weights1.tolist()))
    logs.append("> weights2: {}".format(weights2.tolist()))
    logs.append("> bias_weights1: {}".format(bias_weights1.tolist()))
    logs.append("> bias_weights2: {}".format(bias_weights2.tolist()))
    logs.append("> epsilon: {}, learn_rate: {}, num_input: {}, num_hidden: {}, num_output: {}".format(
        EPSILON, agent.learn_rate, agent.num_input, agent.num_hidden, agent.num_output))

    start = time()
    accuracies_initial = evaluate_agent(agent, reward_system, weights1, weights2, bias_weights1, bias_weights2, EPSILON)
    duration = time() - start
    print("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))
    logs.append("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))

    # ==============================
    # TRAINING AGENT
    # ==============================

    print("\n========== TRAIN AGENT ==========")
    logs.append("\n========== TRAIN AGENT ==========")

    controller = get_controller()
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    controller.loop_episodes(agent)
    logs.append("Agent and plots timestamp: {}".format(timestamp))

    # ==============================
    # EVAL TRAINED AGENT
    # ==============================

    print("\n========== MEASURE ACCURACY (TRAINED) ==========")
    logs.append("\n========== MEASURE ACCURACY (TRAINED) ==========")

    with open(find_agent_file(timestamp)) as agent_file:
        repr_dict = json.load(agent_file)
    representation = (
        repr_dict["weights1"], repr_dict["weights2"], repr_dict["bias_weights1"], repr_dict["bias_weights2"],
        repr_dict["epsilon"], repr_dict["learn_rate"], repr_dict["num_input"], repr_dict["num_hidden"],
        repr_dict["num_output"]
    )
    agent = AgentCorpusQLearning(AgentRepresentation(*representation))
    final_epsilon = repr_dict["epsilon"]
    weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()
    logs.append("Agent representation")
    logs.append("> weights1: {}".format(weights1.tolist()))
    logs.append("> weights2: {}".format(weights2.tolist()))
    logs.append("> bias_weights1: {}".format(bias_weights1.tolist()))
    logs.append("> bias_weights2: {}".format(bias_weights2.tolist()))
    logs.append("> epsilon: {}, learn_rate: {}, num_input: {}, num_hidden: {}, num_output: {}".format(
        final_epsilon, agent.learn_rate, agent.num_input, agent.num_hidden, agent.num_output))

    start = time()
    accuracies_trained = evaluate_agent(agent, reward_system, weights1, weights2, bias_weights1, bias_weights2, final_epsilon)
    duration = time() - start
    print("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))
    logs.append("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))

    # ==============================
    # COMPUTE ACCURACY TABLES
    # ==============================

    print("\n========== ACCURACY TABLE (INITIAL) ==========")
    logs.append("\n========== ACCURACY TABLE (INITIAL) ==========")
    logs = print_accuracy_table(accuracies_initial, logs)

    print("\n========== ACCURACY TABLE (TRAINED) ==========")
    logs.append("\n========== ACCURACY TABLE (TRAINED) ==========")
    logs = print_accuracy_table(accuracies_trained, logs)

    # ==============================
    # WRITE LOGS TO LOG FILE
    # ==============================

    with open(log_file, "w+") as file:
        log_lines = list(map(lambda l: l + "\n", logs))
        file.writelines(log_lines)
finally:
    cleanup_storage()
