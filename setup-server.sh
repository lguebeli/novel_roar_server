#!/bin/bash

sudo python3 setup.py install

virtualenv venv
./venv/Scripts/activate

python3 server.py
