from datetime import datetime
from http import HTTPStatus
import json
import os

from flask import Blueprint, request

fp_bp = Blueprint("fingerprint", __name__, url_prefix="/fp")
storage_path = os.path.abspath(os.path.join(os.curdir, "./fingerprints"))


@fp_bp.route("/<mac>", methods=["POST"])
def report_fingerprint(mac):
    body = json.loads(request.data)

    os.makedirs(storage_path, exist_ok=True)
    fp_path = os.path.join(storage_path, "fp-{time}.txt".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S")))
    with open(fp_path, "x") as file:
        file.write(str(body["fp"]))

    return "", HTTPStatus.NO_CONTENT
