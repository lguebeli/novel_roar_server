from datetime import datetime
from http import HTTPStatus
import json
import os

from flask import Blueprint, request

from environment.state_handling import is_multi_fp_collection, get_fp_path


fp_bp = Blueprint("fingerprint", __name__, url_prefix="/fp")


def write_fingerprint_to_file(fp, storage_path, is_multi):
    os.makedirs(storage_path, exist_ok=True)
    file_name = "fp-{time}.txt".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S")) if is_multi else "fp.txt"
    fp_path = os.path.join(storage_path, file_name)
    with open(fp_path, "x" if is_multi else "w") as file:
        file.write(fp)


@fp_bp.route("/<mac>", methods=["POST"])
def report_fingerprint(mac):
    body = json.loads(request.data)

    write_fingerprint_to_file(fp=str(body["fp"]), storage_path=get_fp_path(), is_multi=is_multi_fp_collection())

    return "", HTTPStatus.CREATED
