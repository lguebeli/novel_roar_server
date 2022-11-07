from http import HTTPStatus

from flask import Blueprint

from environment.state_handling import set_rw_done


rw_bp = Blueprint("ransomware", __name__, url_prefix="/rw")


@rw_bp.route("/done", methods=["PUT"])
def mark_done():
    set_rw_done()
    return "", HTTPStatus.NO_CONTENT
