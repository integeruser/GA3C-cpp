#!/usr/bin/env bash
dirname () { python -c "import os; print(os.path.dirname(os.path.realpath('$0')))"; }
cd "$(dirname "$0")"

while [[ $(pgrep -f gym-uds-server.py) ]] ; do
    pkill -f gym-uds-server.py
    sleep 1
done
