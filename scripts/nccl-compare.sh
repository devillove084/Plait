#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
driver_major=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)
cuda_home=${CUDA_HOME:-/usr/local/cuda}
if [ "${driver_major:-99}" -lt 13 ] && [ -x /opt/cuda129/bin/nvcc ] \
  && { [ -z "${CUDA_HOME:-}" ] || [ "$CUDA_HOME" = "/usr/local/cuda" ]; }; then
  cuda_home=/opt/cuda129
fi
nccl_home=${NCCL_HOME:-$cuda_home}
if [ -z "${NCCL_HOME:-}" ] && [ "$nccl_home" = "/opt/cuda129" ]; then
  conda_nccl=/opt/conda310/lib/python3.10/site-packages/nvidia/nccl
  if [ -f "$conda_nccl/lib/libnccl.so.2" ]; then
    nccl_home=$conda_nccl
  fi
fi
nccl_include=${NCCL_INCLUDE:-"$nccl_home/include"}
nccl_lib_dir=${NCCL_LIB_DIR:-"$nccl_home/lib64"}
if [ ! -e "$nccl_lib_dir/libnccl.so" ] && [ ! -e "$nccl_lib_dir/libnccl.so.2" ]; then
  nccl_lib_dir="$nccl_home/lib"
fi
if [ -n "${NVCC:-}" ]; then
  nvcc=$NVCC
elif [ -x "$cuda_home/bin/nvcc" ]; then
  nvcc="$cuda_home/bin/nvcc"
else
  nvcc=nvcc
fi
gpus=${PLAIT_NCCL_GPUS:-$(nvidia-smi -L | wc -l)}
binary=target/nccl-allreduce
sweep=${1:-}

if [ ! -x "$binary" ] || [ scripts/nccl-allreduce.cu -nt "$binary" ]; then
  mkdir -p target
  link_nccl=(-L"$nccl_lib_dir")
  if [ -e "$nccl_lib_dir/libnccl.so" ]; then
    link_nccl+=(-lnccl)
  else
    link_nccl+=(-Xlinker -l:libnccl.so.2)
  fi
  "$nvcc" -O3 -std=c++17 -I"$nccl_include" scripts/nccl-allreduce.cu \
    "${link_nccl[@]}" -L"$cuda_home/lib64" -L"$cuda_home/lib" -lcudart \
    -Xlinker -rpath,"$nccl_lib_dir" -Xlinker -rpath,"$cuda_home/lib64" \
    -Xlinker -rpath,"$cuda_home/lib" -o "$binary"
fi

echo "== NCCL 12-contribution matrix allreduce =="
nccl_out=$("$binary" --gpus "$gpus" --m "${PLAIT_MATRIX_M:-8}" \
  --n "${PLAIT_MATRIX_N:-8}" --k "${PLAIT_MATRIX_K:-6}" --logical 12 \
  --warmup 5 --iters 20)
echo "$nccl_out"
nccl_sum=$(printf '%s\n' "$nccl_out" | sed -n 's/.*checksum_nccl=\([^ ]*\).*/\1/p')

echo "== Plait 12-replica matrix collective =="
plait_out=$(./scripts/matrix-local.sh 2>&1)
echo "$plait_out"
plait_sum=$(printf '%s\n' "$plait_out" | sed -n 's/.*checksum \([-0-9.]*\).*/\1/p')

if [ -z "$nccl_sum" ] || [ -z "$plait_sum" ]; then
  echo "could not parse checksums (nccl=$nccl_sum plait=$plait_sum)" >&2
  exit 1
fi
awk -v left="$plait_sum" -v right="$nccl_sum" 'BEGIN {
  delta = left - right;
  if (delta < 0) delta = -delta;
  tolerance = 1e-5 * (left + right + 1.0);
  if (delta > tolerance) {
    printf("checksum mismatch: plait=%s nccl=%s\n", left, right) > "/dev/stderr";
    exit 1;
  }
  printf("checksum match: plait=%s nccl=%s\n", left, right);
}' 

if [ "$sweep" = "--sweep" ]; then
  echo "== NCCL allreduce bandwidth sweep =="
  "$binary" --gpus "$gpus" --allreduce --min-bytes 4096 --max-bytes 67108864 \
    --step 4 --warmup 5 --iters 20
fi
