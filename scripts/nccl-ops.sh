#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
nccl_home=${NCCL_HOME:-/usr/local/cuda}
if [ ! -x target/nccl-allreduce ] || [ scripts/nccl-allreduce.cu -nt target/nccl-allreduce ]; then
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -I"$nccl_home/include" \
    scripts/nccl-allreduce.cu -L"$nccl_home/lib64" -lnccl -lcudart \
    -o target/nccl-allreduce
fi
gpus=${PLAIT_NCCL_GPUS:-8}
bytes=${PLAIT_BYTES:-1048576}
for op in allreduce broadcast reduce allgather reducescatter; do
  target/nccl-allreduce --gpus "$gpus" --op "$op" --min-bytes "$bytes" \
    --max-bytes "$bytes" --warmup 5 --iters 20 | tail -1
  cargo run -q --locked -p plait-node -- collectives --gpus "$gpus" \
    --bytes "$bytes" --iters 5 --op "$op" | tail -1
done
