#!/usr/bin/env bash
cd $(dirname $(realpath $0))

./stop-gym-uds-servers.sh

ENV_ID="CartPole-v0"
N=5
for ((i=0; i < $N; ++i)); do
    python third-party/gym-uds-api/gym-uds-server.py $ENV_ID /tmp/gym-uds-socket-GA3C-$i &
done
