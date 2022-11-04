import os

FP_READY = False
RW_DONE = False


def is_fp_ready():
    return FP_READY


def set_fp_ready(ready_state):
    global FP_READY
    FP_READY = ready_state


def is_rw_done():
    return RW_DONE


def set_rw_done():
    global RW_DONE
    RW_DONE = True


def get_fp_path():
    return os.path.abspath(os.path.join(os.curdir, "./fingerprint"))


def collect_fingerprint():
    with open(get_fp_path(), "r") as file:
        fp = file.readline().replace("[", "").replace("]", "").replace(" ", "")
    return fp
