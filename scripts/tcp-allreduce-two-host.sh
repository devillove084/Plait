#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bytes=${PLAIT_TCP_BYTES:-1048576}
iters=${PLAIT_TCP_ITERS:-5}
root_ip=${PLAIT_ROOT_IP:-11.166.42.141}
port=${PLAIT_TCP_PORT:-29600}
remote=${PLAIT_REMOTE:-pro5000}
remote_dir=${PLAIT_REMOTE_DIR:-/root/plait-src}

/opt/cuda129/bin/nvcc -O3 -std=c++17 scripts/tcp-allreduce.cu -o target/tcp-allreduce
tar -czf /tmp/tcp-allreduce-src.tgz scripts/tcp-allreduce.cu
ssh "$remote" "mkdir -p '$remote_dir' && tar -xzf - -C '$remote_dir'" < /tmp/tcp-allreduce-src.tgz
ssh "$remote" "cd '$remote_dir' && /usr/local/cuda/bin/nvcc -O3 -std=c++17 scripts/tcp-allreduce.cu -o target/tcp-allreduce"

rm -f /tmp/tcp-allreduce-local-*.log /tmp/tcp-allreduce-remote.log
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank target/tcp-allreduce "$rank" 12 0 "$bytes" "$iters" "$root_ip" "$port" \
    >"/tmp/tcp-allreduce-local-$rank.log" 2>&1 &
done
ssh "$remote" "cd '$remote_dir' && for i in 0 1 2 3 4 5 6 7; do rank=\$((4+i)); CUDA_VISIBLE_DEVICES=\$i target/tcp-allreduce \"\$rank\" 12 0 '$bytes' '$iters' '$root_ip' '$port' > /tmp/tcp-allreduce-remote-\$rank.log 2>&1 & done; wait; cat /tmp/tcp-allreduce-remote-*.log" > /tmp/tcp-allreduce-remote.log 2>&1 &
wait || true
cat /tmp/tcp-allreduce-local-*.log
cat /tmp/tcp-allreduce-remote.log
