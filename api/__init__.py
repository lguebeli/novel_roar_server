import os

from flask import Flask

from . import status, fingerprint, ransomware


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.register_blueprint(status.status_bp)
    app.register_blueprint(fingerprint.fp_bp)
    app.register_blueprint(ransomware.rw_bp)

    return app
