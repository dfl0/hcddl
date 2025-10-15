#!/usr/bin/env bash

# The client side of a local parameter server communicates with the global
# parameter server. This is handled by a Flower SuperNode.
#
# The following ports are the defaults for `flower-supernode`:
#
#     :9092 - Global parameter server SuperLink Fleet API (gRPC) port (still
#             needs to be specified since we are using a custom address).
#
#     :9094 - Own port used for ClientAppIo API (to communicate with ClientApp running).

client_cmd=(
    flower-supernode
    --insecure
    --superlink $GLB_PS_ADDR:9092
)

# The server side of a local parameter server communicates with the edge
# clients. This is handled by a Flower SuperLink.
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

COL_GRY="\033[0;90m"
COL_CYA="\033[96m"
COL_PUR="\033[95m"
COL_CLR="\033[0m"

CLIENT_TAG=$(printf "${COL_GRY}[${COL_CYA}CLIENT${COL_GRY}]${COL_CLR}")
SERVER_TAG=$(printf "${COL_GRY}[${COL_PUR}SERVER${COL_GRY}]${COL_CLR}")

echo -e "Starting ${COL_CYA}SuperNode${COL_CLR} process..."
# $client_cmd 2>&1 | sed "s/^/${CLIENT_TAG} /" &
PYTHONUNBUFFERED=1 "${client_cmd[@]}" 2>&1 | stdbuf -oL sed "s/^/${CLIENT_TAG} /" &
# "${client_cmd[@]}" &
client_pid=$!

echo -e "Starting ${COL_PUR}SuperLink${COL_CLR} process..."
# $server_cmd 2>&1 | sed "s/^/${SERVER_TAG} /" &
PYTHONUNBUFFERED=1 "${server_cmd[@]}" 2>&1 | stdbuf -oL sed "s/^/${SERVER_TAG} /" &
# "${server_cmd[@]}"  &
server_pid=$!

cleanup() {
    echo -e "\nStopping SuperNode and SuperLink processes..."
    kill $client_pid $server_pid 2>/dev/null
    wait $client_pid $server_pid 2>/dev/null
}

trap cleanup SIGINT SIGTERM

wait
