import os
import signal
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Process
from time import sleep, time

import numpy as np
from tqdm import tqdm

from v23.agent.agent import AgentIdealADSarsaTabular
from v22.agent.agent import AgentSarsaTabular
from api import create_app
from environment.settings import EPSILON_V23, EPSILON_V22
from environment.constructor import get_controller
from environment.reward.abstract_reward import AbstractReward
from environment.settings import EVALUATION_CSV_FOLDER_PATH
from environment.state_handling import initialize_storage, cleanup_storage, set_prototype, get_num_configs, \
    get_storage_path, set_simulation, get_instance_number, setup_child_instance, is_api_running, set_api_running

"""
Want to change evaluated prototype?
1) adjust the known best action (setup) if necessary (depending on AD and reward system)
2) Adjust the filename of the logfile (setup) to match the SARSA prototype settings.
3) Set prototype to evaluated prototype number in setup below (start of try-block).
"""

def start_api(instance_number):
    setup_child_instance(instance_number)
    app = create_app()
    print("==============================\nStart API\n==============================")
    set_api_running()
    app.run(host="0.0.0.0", port=5000)

def kill_process(proc):
    print("kill Process", proc)
    proc.terminate()
    print("killed Process", proc)
    timeout = 10
    start = time()
    while proc.is_alive() and time() - start < timeout:
        sleep(1)
    if proc.is_alive():
        proc.kill()
        print("...we had to put it down", proc)
        sleep(2)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            print("...die already", proc)
        else:
            print(proc, "now dead")
    else:
        print(proc, "now dead")

def evaluate_agent(agent, epsilon):
    accuracies_overall = {"total": 0}
    accuracies_configs = {}
    num_configs = get_num_configs()

    for config in range(num_configs):
        accuracies_overall[config] = 0

    # Eval normal
    print("Normal")
    normal_fp_dir = os.path.join(EVALUATION_CSV_FOLDER_PATH, "normal")
    fp_files = os.listdir(normal_fp_dir)
    accuracies_configs["normal"] = {"total": 0}
    for fp_file in tqdm(fp_files):
        with open(os.path.join(normal_fp_dir, fp_file)) as file:
            fp = file.readline()[1:-1].replace(" ", "")
        state = transform_fp(fp)
        selected_action, q_values = agent.predict(epsilon, state)

        if selected_action not in accuracies_overall:
            accuracies_overall[selected_action] = 1
        else:
            accuracies_overall[selected_action] += 1
        accuracies_overall["total"] += 1

        if selected_action not in accuracies_configs["normal"]:
            accuracies_configs["normal"][selected_action] = 1
        else:
            accuracies_configs["normal"][selected_action] += 1
        accuracies_configs["normal"]["total"] += 1

    # Eval infected
    for config in range(num_configs):
        print("\nConfig", config)
        config_fp_dir = os.path.join(EVALUATION_CSV_FOLDER_PATH, "infected-c{}".format(config))
        fp_files = os.listdir(config_fp_dir)
        accuracies_configs[config] = {"total": 0}
        for fp_file in tqdm(fp_files):
            with open(os.path.join(config_fp_dir, fp_file)) as file:
                fp = file.readline()[1:-1].replace(" ", "")
            state = transform_fp(fp)
            selected_action, q_values = agent.predict(epsilon, state)

            if selected_action not in accuracies_overall:
                accuracies_overall[selected_action] = 1
            else:
                accuracies_overall[selected_action] += 1
            accuracies_overall["total"] += 1

            if selected_action not in accuracies_configs[config]:
                accuracies_configs[config][selected_action] = 1
            else:
                accuracies_configs[config][selected_action] += 1
            accuracies_configs[config]["total"] += 1
    return accuracies_overall, accuracies_configs

def find_agent_file(timestamp):
    storage_path = get_storage_path()
    files = os.listdir(storage_path)
    filtered = [f for f in files if f.startswith(f"{timestamp}=") and f.endswith(".json")]
    if not filtered:
        raise FileNotFoundError(f"No Q-table file found for timestamp {timestamp}")
    return os.path.join(storage_path, filtered.pop())

def transform_fp(fp):
    split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
    return np.asarray(split_to_floats).reshape(-1, 1)  # shape (F, 1)

def print_accuracy_table(accuracies_overall, accuracies_configs, logs):
    num_configs = get_num_configs()
    print("----- Per Config -----")
    logs.append("----- Per Config -----")

    # From normal states
    line = []
    key_keys = sorted([k for k in accuracies_configs["normal"].keys() if not isinstance(k, str)])
    for key in range(num_configs):
        value = accuracies_configs["normal"][key] if key in key_keys else 0
        line.append(f"c{key} {value / accuracies_configs['normal']['total'] * 100:05.2f}% ({value}/{accuracies_configs['normal']['total']})\t")
    print("Normal:\t", *line, sep="\t")
    logs.append("\t".join(["Normal:\t", *line]).strip())

    # From infected states
    config_keys = sorted([k for k in accuracies_configs.keys() if not isinstance(k, str)])
    for config in config_keys:
        line = []
        key_keys = sorted([k for k in accuracies_configs[config].keys() if not isinstance(k, str)])
        for key in range(num_configs):
            value = accuracies_configs[config][key] if key in key_keys else 0
            line.append(f"c{key} {value / accuracies_configs[config]['total'] * 100:05.2f}% ({value}/{accuracies_configs[config]['total']})\t")
        print(f"Config {config}:", *line, sep="\t")
        logs.append("\t".join([f"Config {config}:", *line]).strip())

    print("\n----- Overall -----")
    logs.append("\n----- Overall -----")

    overall_keys = sorted([k for k in accuracies_overall.keys() if not isinstance(k, str)])
    for config in range(num_configs):
        value = accuracies_overall[config] if config in overall_keys else 0
        print(f"Config {config}:\t{value / accuracies_overall['total'] * 100:05.2f}% ({value}/{accuracies_overall['total']})")
        logs.append(f"Config {config}:\t{value / accuracies_overall['total'] * 100:05.2f}% ({value}/{accuracies_overall['total']})")

    return logs

if __name__ == "__main__":
    total_start = time()
    prototype_description = "p23-100e=e0.1d0.9a0.005y0.01=Tabular=IdealAD"  #change the description here
    KNOWN_BEST_ACTION = 3                                                   #change the KBA here

    log_file = os.path.join(os.path.curdir, "storage",
                            f"accuracy-report={datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}={prototype_description}.txt")
    logs = []

    print("========== PREPARE ENVIRONMENT ==========\nAD evaluation is written to log file directly")
    initialize_storage()
    procs = []
    try:
        prototype_num = "23"     # Change this number to switch implementations (22 or 23)
        set_prototype(prototype_num)
        simulated = True
        set_simulation(simulated)
        np.random.seed(42)

        if prototype_num == "22": # Don't change this number!
            AgentClass = AgentSarsaTabular
            eps = EPSILON_V22
        else:
            AgentClass = AgentIdealADSarsaTabular
            eps = EPSILON_V23

        with open(log_file, "w+") as f:
            with redirect_stdout(f):
                print("========== PREPARE ENVIRONMENT ==========")
                AbstractReward.prepare_reward_computation()
        EPSILON = eps

        # Eval untrained agent
        print("\n========== MEASURE ACCURACY (INITIAL) ==========")
        logs.append("\n========== MEASURE ACCURACY (INITIAL) ==========")
        initial_agent = AgentClass()
        print(f"Evaluating agent {initial_agent} with settings {prototype_description}.\n")
        logs.append(f"Evaluating agent {initial_agent} with settings {prototype_description}.\n")
        logs.append("Agent representation")
        logs.append("> prototype: SARSA Tabular")
        logs.append(f"> number of states in Q-table: {len(initial_agent.q_table)}")

        start = time()
        accuracies_initial_overall, accuracies_initial_configs = evaluate_agent(initial_agent, EPSILON)
        duration = time() - start
        print(f"\nEvaluation took {duration:.3f}s, roughly {duration / 60:.1f}min.")
        logs.append(f"\nEvaluation took {duration:.3f}s, roughly {duration / 60:.1f}min.")

        # Training agent
        if not simulated:
            proc_api = Process(target=start_api, args=(get_instance_number(),))
            procs.append(proc_api)
            proc_api.start()
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)

        print("\n========== TRAIN AGENT ==========")
        logs.append("\n========== TRAIN AGENT ==========")
        agent = AgentClass()
        controller = get_controller()
        training_start = time()
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        agent_file, _ = controller.loop_episodes(agent)  # Assumes loop_episodes returns agent_file, all_rewards
        training_duration = time() - training_start
        logs.append(f"Agent file: {agent_file}")
        print(f"Training took {training_duration:.3f}s, roughly {training_duration / 60:.1f}min.")
        logs.append(f"Training took {training_duration:.3f}s, roughly {training_duration / 60:.1f}min.")

        # Eval trained agent
        print("\n========== MEASURE ACCURACY (TRAINED) ==========")
        logs.append("\n========== MEASURE ACCURACY (TRAINED) ==========")
        trained_agent = AgentClass()
        trained_agent.load_q_table(agent_file)
        logs.append("Agent representation")
        logs.append("> prototype: SARSA Tabular")
        logs.append(f"> number of states in Q-table: {len(trained_agent.q_table)}")

        start = time()
        accuracies_trained_overall, accuracies_trained_configs = evaluate_agent(trained_agent, EPSILON)
        duration = time() - start
        print(f"\nEvaluation took {duration:.3f}s, roughly {duration / 60:.1f}min.")
        logs.append(f"\nEvaluation took {duration:.3f}s, roughly {duration / 60:.1f}min.")

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
        known_best_initial = f"{val_initial / accuracies_initial_overall['total'] * 100:05.2f}% ({val_initial}/{accuracies_initial_overall['total']})"
        val_trained = accuracies_trained_overall.get(KNOWN_BEST_ACTION, 0)
        known_best_trained = f"{val_trained / accuracies_trained_overall['total'] * 100:05.2f}% ({val_trained}/{accuracies_trained_overall['total']})"
        print(f"For known best action {KNOWN_BEST_ACTION}: from {known_best_initial} to {known_best_trained}.")
        logs.append(f"For known best action {KNOWN_BEST_ACTION}: from {known_best_initial} to {known_best_trained}.")

        total_duration = time() - total_start
        print(f"Accuracy computation took {total_duration:.3f}s in total, roughly {total_duration / 60:.1f}min.")
        logs.append(f"Accuracy computation took {total_duration:.3f}s in total, roughly {total_duration / 60:.1f}min.")

        with open(log_file, "a") as file:
            log_lines = [l + "\n" for l in logs]
            file.writelines(log_lines)
    finally:
        for proc in procs:
            kill_process(proc)
        print("- Parallel processes killed.")
        cleanup_storage()
        print("- Storage cleaned up.\n==============================")