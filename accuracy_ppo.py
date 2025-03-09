import json
import os
import signal
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Process
from time import sleep, time

import numpy as np
from tqdm import tqdm

from v23.agent.agent import AgentPPO
from v23.environment.controller import ControllerPPO
from environment.constructor import get_controller
from environment.reward.abstract_reward import AbstractReward
from environment.settings import EVALUATION_CSV_FOLDER_PATH
from environment.state_handling import initialize_storage, cleanup_storage, set_prototype, get_num_configs, \
    get_storage_path, set_simulation, get_instance_number, setup_child_instance, is_api_running, set_api_running

# Placeholder for Flask app creation (replace with your actual Flask app import)
def create_app():
    from flask import Flask
    app = Flask(__name__)
    return app

def start_api(instance_number):
    setup_child_instance(instance_number)
    app = create_app()
    print("==============================\nStart API\n==============================")
    set_api_running()
    app.run(host="0.0.0.0", port=5000)

def kill_process(proc):
    print("kill Process", proc)
    proc.terminate()
    timeout = 10
    start = time()
    while proc.is_alive() and time() - start < timeout:
        sleep(1)
    if proc.is_alive():
        proc.kill()
        sleep(2)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            print("...die already", proc)
        else:
            print(proc, "now dead")
    else:
        print(proc, "now dead")

def transform_fp(fp):
    """Convert fingerprint string to NumPy array, matching controller.py."""
    return np.array(list(map(float, fp.split(","))))  # Shape: (F,)

def evaluate_agent(agent):
    """Evaluate the agent's accuracy on normal and infected fingerprints."""
    accuracies_overall = {"total": 0}
    accuracies_configs = {}
    num_configs = get_num_configs()

    for config in range(num_configs):
        accuracies_overall[config] = 0

    # Evaluate normal fingerprints
    print("Normal")
    normal_fp_dir = os.path.join(EVALUATION_CSV_FOLDER_PATH, "normal")
    fp_files = os.listdir(normal_fp_dir)
    accuracies_configs["normal"] = {"total": 0}
    for fp_file in tqdm(fp_files):
        with open(os.path.join(normal_fp_dir, fp_file)) as file:
            fp = file.readline()[1:-1].replace(" ", "")
        state = transform_fp(fp)  # Shape: (F,)
        selected_action = agent.evaluate_action(state)

        accuracies_overall[selected_action] = accuracies_overall.get(selected_action, 0) + 1
        accuracies_overall["total"] += 1

        accuracies_configs["normal"][selected_action] = accuracies_configs["normal"].get(selected_action, 0) + 1
        accuracies_configs["normal"]["total"] += 1

    # Evaluate infected fingerprints
    for config in range(num_configs):
        print(f"\nConfig {config}")
        config_fp_dir = os.path.join(EVALUATION_CSV_FOLDER_PATH, f"infected-c{config}")
        fp_files = os.listdir(config_fp_dir)
        accuracies_configs[config] = {"total": 0}
        for fp_file in tqdm(fp_files):
            with open(os.path.join(config_fp_dir, fp_file)) as file:
                fp = file.readline()[1:-1].replace(" ", "")
            state = transform_fp(fp)  # Shape: (F,)
            selected_action = agent.evaluate_action(state)

            accuracies_overall[selected_action] = accuracies_overall.get(selected_action, 0) + 1
            accuracies_overall["total"] += 1

            accuracies_configs[config][selected_action] = accuracies_configs[config].get(selected_action, 0) + 1
            accuracies_configs[config]["total"] += 1

    return accuracies_overall, accuracies_configs

def find_agent_file(timestamp):
    """Find the saved agent representation file."""
    storage_path = get_storage_path()
    files = os.listdir(storage_path)
    filtered = [f for f in files if f.startswith(f"agent={timestamp}")]
    if not filtered:
        raise FileNotFoundError(f"No agent file found for timestamp {timestamp}")
    return os.path.join(storage_path, filtered[0])

def print_accuracy_table(accuracies_overall, accuracies_configs, logs):
    """Print and log accuracy tables."""
    num_configs = get_num_configs()
    print("----- Per Config -----")
    logs.append("----- Per Config -----")

    # from normal states
    line = []
    key_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_configs["normal"].keys()))))
    for key in range(num_configs):
        value = accuracies_configs["normal"][key] if key in key_keys else 0
        line.append("c{} {}% ({}/{})\t".format(key, "%05.2f" % (value / accuracies_configs["normal"]["total"] * 100),
                                               value, accuracies_configs["normal"]["total"]))
    print("Normal:\t", *line, sep="\t")
    logs.append("\t".join(["Normal:\t", *line]).strip())

    # from infected states
    config_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_configs.keys()))))
    for config in config_keys:
        line = []
        key_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_configs[config].keys()))))
        for key in range(num_configs):
            value = accuracies_configs[config][key] if key in key_keys else 0
            line.append("c{} {}% ({}/{})\t".format(key, "%05.2f" % (value / accuracies_configs[config]["total"] * 100),
                                                   value, accuracies_configs[config]["total"]))
        print("Config {}:".format(config), *line, sep="\t")
        logs.append("\t".join(["Config {}:".format(config), *line]).strip())

    print("\n----- Overall -----")
    logs.append("\n----- Overall -----")

    overall_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_overall.keys()))))
    for config in range(num_configs):
        value = accuracies_overall[config] if config in overall_keys else 0
        print("Config {}:\t{}% ({}/{})".format(config, "%05.2f" % (value / accuracies_overall["total"] * 100), value,
                                               accuracies_overall["total"]))
        logs.append("Config {}:\t{}% ({}/{})".format(config, "%05.2f" % (value / accuracies_overall["total"] * 100),
                                                     value, accuracies_overall["total"]))

    return logs

if __name__ == "__main__":
    # ==============================
    # SETUP
    # ==============================
    total_start = time()
    prototype_description = "p23-2000e=learn_rate0.001-clip_epsilon0.2-gamma0.99-lambda0.95"
    KNOWN_BEST_ACTION = 3

    log_file = os.path.join(os.path.curdir, "storage",
                            f"accuracy-report={datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}={prototype_description}.txt")
    logs = []

    print("========== PREPARE ENVIRONMENT ==========\nAD evaluation is written to log file directly")
    initialize_storage()
    procs = []
    try:
        set_prototype("23")
        simulated = True  # Set to False for real environment
        set_simulation(simulated)
        np.random.seed(42)

        # Write initial log
        with open(log_file, "w+") as f:
            with redirect_stdout(f):
                print("========== PREPARE ENVIRONMENT ==========")
                AbstractReward.prepare_reward_computation()

        # Evaluate untrained agent
        print("\n========== MEASURE ACCURACY (INITIAL) ==========")
        logs.append("\n========== MEASURE ACCURACY (INITIAL) ==========")

        agent = AgentPPO()
        print(f"Evaluating agent {agent} with settings {prototype_description}.\n")
        logs.append(f"Evaluating agent {agent} with settings {prototype_description}.\n")
        logs.append("Agent representation")
        logs.append("> prototype: PPO")
        logs.append(f"> weights1: {agent.weights1.tolist()}")
        logs.append(f"> weights_policy: {agent.weights_policy.tolist()}")
        logs.append(f"> weights_value: {agent.weights_value.tolist()}")
        logs.append(f"> learn_rate: {agent.learn_rate}, clip_epsilon: {agent.clip_epsilon}, "
                    f"gamma: {agent.gamma}, lambda_: {agent.lambda_}, value_coef: {agent.value_coef}, "
                    f"entropy_coef: {agent.entropy_coef}, epochs: {agent.epochs}, batch_size: {agent.batch_size}")

        start = time()
        accuracies_initial_overall, accuracies_initial_configs = evaluate_agent(agent)
        duration = time() - start
        print(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))
        logs.append(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))

        # ==============================
        # TRAINING AGENT
        # ==============================

        if not simulated:
            proc_api = Process(target=start_api, args=(get_instance_number(),))
            procs.append(proc_api)
            proc_api.start()
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)

        print("\n========== TRAIN AGENT ==========")
        logs.append("\n========== TRAIN AGENT ==========")

        controller = ControllerPPO()
        training_start = time()
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        controller.run_c2(agent)  # Train the agent
        training_duration = time() - training_start
        logs.append(f"Agent and plots timestamp: {timestamp}")
        print(f"Training took %.3fs, roughly %.1fmin." % (training_duration, training_duration / 60))
        logs.append(f"Training took %.3fs, roughly %.1fmin." % (training_duration, training_duration / 60))

        # ==============================
        # EVAL TRAINED AGENT
        # ==============================

        print("\n========== MEASURE ACCURACY (TRAINED) ==========")
        logs.append("\n========== MEASURE ACCURACY (TRAINED) ==========")

        with open(find_agent_file(timestamp), "r") as agent_file:
            repr_dict = json.load(agent_file)
        agent = AgentPPO(representation=repr_dict)

        logs.append("Trained agent representation")
        logs.append("> prototype: PPO")
        logs.append(f"> weights1: {agent.weights1.tolist()}")
        logs.append(f"> weights_policy: {agent.weights_policy.tolist()}")
        logs.append(f"> weights_value: {agent.weights_value.tolist()}")
        logs.append(f"> learn_rate: {agent.learn_rate}, clip_epsilon: {agent.clip_epsilon}, "
                    f"gamma: {agent.gamma}, lambda_: {agent.lambda_}, value_coef: {agent.value_coef}, "
                    f"entropy_coef: {agent.entropy_coef}, epochs: {agent.epochs}, batch_size: {agent.batch_size}")

        start = time()
        accuracies_trained_overall, accuracies_trained_configs = evaluate_agent(agent)
        duration = time() - start
        print(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))
        logs.append(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))

        # Compute accuracy tables
        print("\n========== ACCURACY TABLE (INITIAL) ==========")
        logs.append("\n========== ACCURACY TABLE (INITIAL) ==========")
        logs = print_accuracy_table(accuracies_initial_overall, accuracies_initial_configs, logs)

        print("\n========== ACCURACY TABLE (TRAINED) ==========")
        logs.append("\n========== ACCURACY TABLE (TRAINED) ==========")
        logs = print_accuracy_table(accuracies_trained_overall, accuracies_trained_configs, logs)

        # Show results
        print("\n========== RESULTS ==========")
        logs.append("\n========== RESULTS ==========")

        val_initial = accuracies_initial_overall.get(KNOWN_BEST_ACTION, 0)
        #known_best_initial = f"%05.2f%% ({val_initial}/{accuracies_initial_overall['total']})"
        known_best_initial = "{}% ({}/{})".format("%05.2f" % (val_initial / accuracies_initial_overall["total"] * 100),
                                                  val_initial, accuracies_initial_overall["total"])

        val_trained = accuracies_trained_overall.get(KNOWN_BEST_ACTION, 0)
        #known_best_trained = f"%05.2f%% ({val_trained}/{accuracies_trained_overall['total']})"
        known_best_trained = "{}% ({}/{})".format("%05.2f" % (val_trained / accuracies_initial_overall["total"] * 100),
                                                  val_trained, accuracies_initial_overall["total"])
        print(f"For known best action {KNOWN_BEST_ACTION}: from {known_best_initial} to {known_best_trained}.")
        logs.append(f"For known best action {KNOWN_BEST_ACTION}: from {known_best_initial} to {known_best_trained}.")

        total_duration = time() - total_start
        print(f"Accuracy computation took %.3fs in total, roughly %.1fmin." % (total_duration, total_duration / 60))
        logs.append(f"Accuracy computation took %.3fs in total, roughly %.1fmin." % (total_duration, total_duration / 60))

        # Write logs
        with open(log_file, "a") as file:
            file.writelines([l + "\n" for l in logs])

    finally:
        for proc in procs:
            kill_process(proc)
        print("- Parallel processes killed.")
        cleanup_storage()
        print("- Storage cleaned up.\n==============================")