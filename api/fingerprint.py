import json
from http import HTTPStatus

from flask import Blueprint, request

from environment.state_handling import get_storage_path, is_multi_fp_collection
from utilities.metrics import write_metrics_to_file

fp_bp = Blueprint("fingerprint", __name__, url_prefix="/fp")


@fp_bp.route("/<mac>", methods=["POST"])
def report_fingerprint(mac):
    body = json.loads(request.data)

    write_metrics_to_file(str(body["rate"]), str(body["fp"]), get_storage_path(), is_multi_fp_collection())

    return "", HTTPStatus.CREATED
