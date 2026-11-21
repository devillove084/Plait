#!/bin/sh
# Usage: . python/plait_vllm_env.sh
# Build the PyO3 extension first (maturin develop --release --features python).
export PYTHONPATH="$(pwd)/python${PYTHONPATH:+:$PYTHONPATH}"
export PLAIT_NCCL_MASTER_ADDR="${PLAIT_NCCL_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
