#!/usr/bin/env bash
pkill -f gym-uds-server.py
sleep 1

GYM_UDS_API_DIR="/opt/gym-uds-api"
ENV_ID="CartPole-v0"
N=5
for ((i=0; i < $N; ++i)); do
   python3 $GYM_UDS_API_DIR/gym-uds-server.py $ENV_ID /tmp/gym-uds-socket-GA3C-$i &
done
