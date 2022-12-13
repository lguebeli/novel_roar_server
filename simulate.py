import os
import random

from environment.settings import CSV_FOLDER_PATH, RPI_MODEL_PREFIX
from environment.state_handling import get_fp_path, set_rw_done
from utilities import write_fingerprint_to_file


# ==============================
# SIMULATE CLIENT BEHAVIOR
# ==============================

def simulate_sending_fp(config):
    config_fp_dir = os.path.join(CSV_FOLDER_PATH, RPI_MODEL_PREFIX + "infected-c{}".format(config))
    fp_files = os.listdir(config_fp_dir)
    with open(os.path.join(config_fp_dir, random.choice(fp_files))) as file:
        # print("SIM:", file.name)
        fp = file.read()
    write_fingerprint_to_file(fp=fp, storage_path=get_fp_path(), is_multi=False)


def simulate_sending_rw_done():
    set_rw_done()
