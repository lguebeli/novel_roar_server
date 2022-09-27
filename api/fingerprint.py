from http import HTTPStatus
import json

from flask import Blueprint, request

fp_bp = Blueprint('fingerprint', __name__, url_prefix='/fp')


@fp_bp.route("/", methods=["POST"])
def report_fingerprint():
    body = json.loads(request.data)
    print(body)
    return "", HTTPStatus.NO_CONTENT
