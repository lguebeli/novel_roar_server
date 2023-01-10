import os
from datetime import datetime

from environment.state_handling import set_fp_ready, get_fp_file_path, get_rate_file_path


def write_metrics_to_file(rate, fp, storage_path, is_multi):
    # print("SIM: writing rate/fp", rate, fp, is_multi)
    if not is_multi:
        __write_rate_to_file(rate, storage_path)
    __write_fingerprint_to_file(fp, storage_path, is_multi)
    set_fp_ready(True)


def __write_rate_to_file(rate, storage_path):
    os.makedirs(storage_path, exist_ok=True)
    rate_path = get_rate_file_path()
    with open(rate_path, "w") as file:
        file.write(str(rate))


def __write_fingerprint_to_file(fp, storage_path, is_multi):
    os.makedirs(storage_path, exist_ok=True)
    if is_multi:
        file_name = "fp-{time}.txt".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        fp_path = os.path.join(storage_path, file_name)
    else:
        fp_path = get_fp_file_path()
    with open(fp_path, "x" if is_multi else "w") as file:
        file.write(fp)
