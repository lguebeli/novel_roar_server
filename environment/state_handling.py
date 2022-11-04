import os

COLLECT_MULTIPLE_FP = False
FP_READY = False
RW_DONE = False


def is_multi_fp_collection():
    return COLLECT_MULTIPLE_FP


def set_multi_fp_collection(is_multi):
    global COLLECT_MULTIPLE_FP
    COLLECT_MULTIPLE_FP = is_multi


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
    dir_name = "./fingerprints" if is_multi_fp_collection() else "./fingerprint"
    return os.path.abspath(os.path.join(os.curdir, dir_name))


def collect_fingerprint():
    with open(get_fp_path(), "r") as file:
        fp = file.readline().replace("[", "").replace("]", "").replace(" ", "")
    return fp
