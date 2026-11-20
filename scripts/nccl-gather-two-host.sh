#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bytes=${PLAIT_TCP_BYTES:-1048576}
iters=${PLAIT_TCP_ITERS:-5}
root_ip=${PLAIT_ROOT_IP:-11.166.42.141}
port=${PLAIT_TCP_PORT:-29500}
remote=${PLAIT_REMOTE:-pro5000}
remote_dir=${PLAIT_REMOTE_DIR:-/root/plait-src}
local_nccl=${PLAIT_LOCAL_NCCL:-/tmp/nccl228/internal/extracted/nvidia/nccl/lib}
local_cuda=/opt/cuda129/lib
remote_nccl=/opt/conda310/lib/python3.10/site-packages/nvidia/nccl/lib
remote_include=/opt/conda310/lib/python3.10/site-packages/nvidia/nccl/include
remote_cuda=/opt/conda310/lib/python3.10/site-packages/nvidia/cu13/lib

/opt/cuda129/bin/nvcc -O3 -std=c++17 -I"${local_nccl%/lib}/include" \
  scripts/nccl-gather.cu -L"$local_nccl" -l:libnccl.so.2 -L"$local_cuda" -lcudart \
  -o target/nccl-gather

tar -czf /tmp/nccl-gather-src.tgz scripts/nccl-gather.cu
ssh "$remote" "mkdir -p '$remote_dir' && tar -xzf - -C '$remote_dir'" < /tmp/nccl-gather-src.tgz
ssh "$remote" "cd '$remote_dir' && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -I'$remote_include' scripts/nccl-gather.cu -L'$remote_nccl' -l:libnccl.so.2 -L'$remote_cuda' -lcudart -o target/nccl-gather"

export LD_LIBRARY_PATH="$local_nccl:/opt/cuda129/lib:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

rm -f /tmp/nccl-gather-local-*.log /tmp/nccl-gather-remote.log
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank target/nccl-gather "$rank" 12 0 "$bytes" "$iters" "$root_ip" "$port" \
    >"/tmp/nccl-gather-local-$rank.log" 2>&1 &
done
ssh "$remote" "cd '$remote_dir' && export LD_LIBRARY_PATH='$remote_nccl:$remote_cuda':\${LD_LIBRARY_PATH:-} NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN; for i in 0 1 2 3 4 5 6 7; do rank=\$((4+i)); CUDA_VISIBLE_DEVICES=\$i target/nccl-gather \"\$rank\" 12 0 '$bytes' '$iters' '$root_ip' '$port' > /tmp/nccl-gather-remote-\$rank.log 2>&1 & done; wait; cat /tmp/nccl-gather-remote-*.log" > /tmp/nccl-gather-remote.log 2>&1 &
wait || true
cat /tmp/nccl-gather-local-*.log
cat /tmp/nccl-gather-remote.log
