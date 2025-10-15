#!/usr/bin/env bash

# The global parameter server communicates with the client side of the local
# parameter server. This is handled by a Flower SuperLink.
#
# The following ports are the defaults for `flower-superlink`:
#
#     :9091 - Own port used for ServerAppIo API (to communicate with ServerApp running).
#
#     :9092 - Own port used for Fleet API (to communicate with the local parameter server).
#
#     :9093 - Own port used for Exec API (to receive communications from the FlowerCLI).

server_cmd=(
    flower-superlink
    --insecure
)

set -e

echo -e "Starting Global Parameter Server SuperLink process..."
"${server_cmd[@]}"
