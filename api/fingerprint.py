import json
from http import HTTPStatus

from flask import Blueprint, request

from environment.state_handling import is_multi_fp_collection, get_fp_path
from utilities import write_fingerprint_to_file

fp_bp = Blueprint("fingerprint", __name__, url_prefix="/fp")


@fp_bp.route("/<mac>", methods=["POST"])
def report_fingerprint(mac):
    body = json.loads(request.data)

    write_fingerprint_to_file(fp=str(body["fp"]), storage_path=get_fp_path(), is_multi=is_multi_fp_collection())

    return "", HTTPStatus.CREATED
