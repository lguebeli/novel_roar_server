import os
import random

from environment.settings import CSV_FOLDER_PATH
from environment.state_handling import get_storage_path, is_multi_fp_collection, set_rw_done
from utilities.metrics import write_metrics_to_file


# ==============================
# SIMULATE CLIENT BEHAVIOR
# ==============================

def simulate_sending_fp(config):
    metrics_dir = os.path.join(CSV_FOLDER_PATH, "metrics")
    with open(os.path.join(metrics_dir, "metrics-c{}.txt".format(config))) as metrics_file:
        headers = metrics_file.readline().split(",")
        rate_idx = headers.index("burst_current_rate")
        # print("SIM: rate idx", config, rate_idx)
        metrics_lines = metrics_file.read().split("\n")
        metrics_lines.remove("")
        metrics = list(map(lambda metric_line: metric_line.split(","), metrics_lines))
        # print("SIM: metrics", metrics)
        metric = random.choice(metrics)
        # print("SIM: metric", metric)
        rate = float(metric[rate_idx])
        # print("SIM: rate", rate)

    config_fp_dir = os.path.join(CSV_FOLDER_PATH, "infected-c{}".format(config))
    fp_files = os.listdir(config_fp_dir)
    with open(os.path.join(config_fp_dir, random.choice(fp_files))) as fp_file:
        # print("SIM: fp", fp_file.name)
        fp = fp_file.read()

    write_metrics_to_file(rate, fp, get_storage_path(), is_multi_fp_collection())


def simulate_sending_rw_done():
    set_rw_done()
