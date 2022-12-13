import os
from datetime import datetime

from environment.state_handling import set_fp_ready


def write_metrics_to_file(rate, fp, storage_path, is_multi):
    __write_rate_to_file(rate, storage_path)
    __write_fingerprint_to_file(fp, storage_path, is_multi)
    set_fp_ready(True)


def __write_rate_to_file(rate, storage_path):
    os.makedirs(storage_path, exist_ok=True)
    file_name = "rate.txt"
    fp_path = os.path.join(storage_path, file_name)
    with open(fp_path, "w") as file:
        file.write(str(rate))


def __write_fingerprint_to_file(fp, storage_path, is_multi):
    os.makedirs(storage_path, exist_ok=True)
    file_name = "fp-{time}.txt".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S")) if is_multi else "fp.txt"
    fp_path = os.path.join(storage_path, file_name)
    with open(fp_path, "x" if is_multi else "w") as file:
        file.write(fp)
