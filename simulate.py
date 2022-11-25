import os
import random
from requests import post, put

FP_DIR = os.path.abspath("<local-path-to-collected-fingerprints-folders>")
RPI_MODEL_PREFIX = "mod4-"


def simulate_sending_fp(config):
    config_fp_dir = os.path.join(FP_DIR, RPI_MODEL_PREFIX + "infected-c{}".format(config))
    fp_files = os.listdir(config_fp_dir)
    with open(os.path.join(config_fp_dir, random.choice(fp_files))) as file:
        # print("SIM:", file.name)
        fp = file.read()
    post(url="http://127.0.0.1:5000/fp/somemac", json={"fp": fp})


def simulate_sending_rw_done():
    put(url="http://127.0.0.1:5000/rw/done")
