
NODE_TYPE=${1}
# Validate node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

HEAD_NODE_ADDRESS="10.213.75.203"

RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
export VLLM_HOST_IP="10.213.75.203"
export NCCL_SOCKET_IFNAME=xgbe1
export NCCL_DEBUG=INFO
    RAY_START_CMD+=" --head --port=6379"

else

export VLLM_HOST_IP="10.213.75.158"
export NCCL_SOCKET_IFNAME=xgbe1
export NCCL_DEBUG=INFO
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

/bin/bash -c "${RAY_START_CMD}"