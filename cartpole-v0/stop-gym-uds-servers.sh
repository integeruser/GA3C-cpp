#!/usr/bin/env bash
cd $(dirname $(realpath $0))

while [[ $(pgrep -f gym-uds-server.py) ]] ; do
    pkill -f gym-uds-server.py
    sleep 1
done
