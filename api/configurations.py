import json
import os
import socket

CLIENT_IP = "<Client-IP>"


def send_config(config):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((CLIENT_IP, 42666))
        sock.sendall(bytes(config, encoding="utf-8"))
        print("Sent config", config)


def map_to_ransomware_configuration(action):
    nr_of_configs = len(os.listdir(os.path.join(os.path.abspath(os.path.curdir), "rw-configs")))
    assert 0 <= action < nr_of_configs
    with open(os.path.join(os.path.curdir, "./rw-configs/config-{act}.json".format(act=action)), "r") as conf_file:
        config = json.loads(conf_file.read())
    return config
