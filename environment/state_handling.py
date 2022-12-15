import os

from tinydb import TinyDB, Query
from tinydb.operations import set


# ==============================
# EXECUTION
# ==============================
STORAGE_FOLDER_NAME = "storage"
MULTI_FP_COLLECTION_FOLDER_NAME = "fingerprints"


def is_fp_ready():
    return __query_key("FP_READY")


def set_fp_ready(ready_state):
    __set_value("FP_READY", ready_state)


def is_rw_done():
    return __query_key("RW_DONE")


def set_rw_done(done=True):
    __set_value("RW_DONE", done)


def get_num_configs():
    return len(os.listdir(os.path.join(os.path.abspath(os.path.curdir), "rw-configs")))


def collect_fingerprint():
    with open(os.path.join(get_storage_path(), "fp.txt"), "r") as file:
        fp = file.readline()[1:-1].replace(" ", "")
    # print("Collected FP.")
    return fp


def collect_rate():
    with open(os.path.join(get_storage_path(), "rate.txt"), "r") as file:
        rate = float(file.readline())
    # print("Collected rate.")
    return rate


# ==============================
# ORCHESTRATION
# ==============================
def is_multi_fp_collection():
    return __query_key("COLLECT_MULTIPLE_FP")


def set_multi_fp_collection(is_multi):
    __set_value("COLLECT_MULTIPLE_FP", is_multi)


def get_prototype():
    return __query_key("PROTOTYPE")


def set_prototype(proto):
    __set_value("PROTOTYPE", proto)


def is_simulation():
    return __query_key("SIMULATION")


def set_simulation(simulated):
    __set_value("SIMULATION", simulated)


def is_api_running():
    return __query_key("API")


def set_api_running():
    __set_value("API", True)


# ==============================
# STATE STORAGE
# ==============================
def initialize_storage():
    __prepare_storage_file()
    db = __get_storage()
    db.drop_tables()  # reset database to start from scratch
    db.insert({"key": "COLLECT_MULTIPLE_FP", "value": False})
    db.insert({"key": "FP_READY", "value": False})
    db.insert({"key": "RW_DONE", "value": False})
    db.insert({"key": "PROTOTYPE", "value": 0})
    db.insert({"key": "SIMULATION", "value": False})
    db.insert({"key": "API", "value": False})
    print("Storage ready.")


def get_storage_path():
    dir_name = MULTI_FP_COLLECTION_FOLDER_NAME if is_multi_fp_collection() else STORAGE_FOLDER_NAME
    return os.path.join(os.path.abspath(os.path.curdir), dir_name)


def __get_storage_file_path():
    storage_folder = os.path.join(os.path.abspath(os.path.curdir), STORAGE_FOLDER_NAME)
    storage_file = os.path.join(storage_folder, "storage.json")
    return storage_file, storage_folder


def __get_storage():
    storage_path, _ = __get_storage_file_path()
    return TinyDB(storage_path)


def __prepare_storage_file():
    storage_file, storage_folder = __get_storage_file_path()
    os.makedirs(storage_folder, exist_ok=True)
    with open(storage_file, "w+"):  # create file if not exists and truncate contents if exists
        pass


def __query_key(key):
    flag = __get_storage().get(Query().key == str(key))
    assert flag is not None
    # print("{} is {}".format(key, flag["value"]))
    return flag["value"]


def __set_value(key, value):
    __get_storage().update(set("value", value), Query().key == str(key))
    # print("Set {} to {}".format(key, value))
