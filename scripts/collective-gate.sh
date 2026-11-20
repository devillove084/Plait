#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
nccl_home=${NCCL_HOME:-/usr/local/cuda}
gpus=${PLAIT_NCCL_GPUS:-8}
iters=${PLAIT_GATE_ITERS:-2}
sizes=${PLAIT_GATE_SIZES:-"4096 65536 1048576 16777216"}
ops="allreduce broadcast reduce allgather reducescatter alltoall"
artifact=${PLAIT_GATE_JSON:-target/collective-gate.json}
mkdir -p target

if [ ! -x target/nccl-allreduce ] || [ scripts/nccl-allreduce.cu -nt target/nccl-allreduce ]; then
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -I"$nccl_home/include" \
    scripts/nccl-allreduce.cu -L"$nccl_home/lib64" -lnccl -lcudart \
    -o target/nccl-allreduce
fi

cargo build -q -p plait-node
python3_bin=$(command -v python3 || true)
echo "{" > "$artifact"
first_op=1
for op in $ops; do
  if [ "$first_op" -eq 1 ]; then first_op=0; else echo "," >> "$artifact"; fi
  echo "\"$op\":[" >> "$artifact"
  first_size=1
  for size in $sizes; do
    nccl=$(target/nccl-allreduce --gpus "$gpus" --op "$op" --min-bytes "$size" \
      --max-bytes "$size" --warmup 1 --iters "$iters" | tail -1)
    plait=$(cargo run -q --locked -p plait-node -- collectives --gpus "$gpus" \
      --bytes "$size" --iters "$iters" --op "$op" \
      --trace "target/plait-${op}-${size}.json" | grep 'busbw_GB/s=' | tail -1)
    if ! printf '%s' "$nccl" | grep -q 'errors=0'; then
      echo "NCCL correctness failure: $op $size: $nccl" >&2
      exit 1
    fi
    if ! printf '%s' "$plait" | grep -q 'busbw_GB/s='; then
      echo "Plait failure: $op $size: $plait" >&2
      exit 1
    fi
    if [ -n "${PLAIT_GATE_MIN_RATIO:-}" ]; then
      nccl_bw=$(printf '%s' "$nccl" | sed -n 's/.*busbw_GB\/s=\([0-9.]*\).*/\1/p')
      plait_bw=$(printf '%s' "$plait" | sed -n 's/.*busbw_GB\/s=\([0-9.]*\).*/\1/p')
      if ! awk -v a="$plait_bw" -v b="$nccl_bw" -v r="$PLAIT_GATE_MIN_RATIO" 'BEGIN { exit !(b > 0 && a / b >= r) }'; then
        echo "performance gate failure: $op $size plait=${plait_bw} nccl=${nccl_bw} ratio<$PLAIT_GATE_MIN_RATIO" >&2
        exit 1
      fi
    fi
    if [ "$first_size" -eq 1 ]; then first_size=0; else echo "," >> "$artifact"; fi
    printf '{"size":%s,"nccl":"%s","plait":"%s"}' "$size" \
      "${nccl//\"/\\\"}" "${plait//\"/\\\"}" >> "$artifact"
    echo "gate: $op $size"
    echo "  nccl:  $nccl"
    echo "  plait: $plait"
  done
  echo "]" >> "$artifact"
done
echo "}" >> "$artifact"
echo "collective gate artifact: $artifact"
