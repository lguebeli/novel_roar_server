import json
import os
import random

from environment.settings import TRAINING_CSV_FOLDER_PATH
from environment.state_handling import get_storage_path, is_multi_fp_collection, set_rw_done
from utilities.metrics import write_metrics_to_file


# ==============================
# SIMULATE CLIENT BEHAVIOR
# ==============================

UNLIMITED_CONFIGURATIONS = [1, 2]


def __find_average_rate(config):
    metrics_dir = os.path.join(TRAINING_CSV_FOLDER_PATH, "metrics")
    with open(os.path.join(metrics_dir, "metrics-c{}.txt".format(config)), "r") as file:
        lines = file.readlines()[1:]  # drop headers
        sum = 0
        for line in lines:
            sum += float(line.split(",")[-2])
        avg = sum / len(lines)
        # print("SIM: config", config, "avg %.3f" % avg, "for", sum, "out of", len(lines), "lines")
    return avg


def simulate_sending_fp(config_num):
    config_dir = os.path.join(os.curdir, "rw-configs")
    if config_num in UNLIMITED_CONFIGURATIONS:  # config defines a rate of 0, so we need to collect it from metrics
        rate = __find_average_rate(config_num)
    else:
        with open(os.path.join(config_dir, "config-{}.json".format(config_num)), "r") as config_file:
            config = json.load(config_file)
            rate = int(config["rate"])

    config_fp_dir = os.path.join(TRAINING_CSV_FOLDER_PATH, "infected-c{}".format(config_num))
    fp_files = os.listdir(config_fp_dir)
    with open(os.path.join(config_fp_dir, random.choice(fp_files))) as fp_file:
        # print("SIM: fp", fp_file.name)
        fp = fp_file.read()

    write_metrics_to_file(rate, fp, get_storage_path(), is_multi_fp_collection())


def simulate_sending_rw_done():
    set_rw_done()
