import os
from datetime import datetime

from environment.state_handling import set_fp_ready


def write_fingerprint_to_file(fp, storage_path, is_multi):
    os.makedirs(storage_path, exist_ok=True)
    file_name = "fp-{time}.txt".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S")) if is_multi else "fp.txt"
    fp_path = os.path.join(storage_path, file_name)
    with open(fp_path, "x" if is_multi else "w") as file:
        file.write(fp)
    set_fp_ready(True)
