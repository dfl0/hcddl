#!/usr/bin/env bash

# A worker communicates with the server side of a local parameter server. This
# is handled by a Flower SuperNode.
#
# The following ports are the defaults for `flower-supernode`:
#
#     :9092 - Local parameter server SuperLink Fleet API (gRPC) port (still
#            needs to be specified since we are using a custom address).
#
#     :9094 - Own port used for ClientAppIo API (to communicate with ClientApp running).

client_cmd=(
    flower-supernode
    --insecure
    --superlink $LOC_PS_ADDR:9092
    --node-config "'num-partitions'=$NUM_WORKERS 'partition-id'=$WORKER_ID"
)

set -e

echo -e "Starting Worker SuperNode process..."
"${client_cmd[@]}"
