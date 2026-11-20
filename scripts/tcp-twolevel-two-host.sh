#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bytes=${PLAIT_TCP_BYTES:-1048576}
iters=${PLAIT_TCP_ITERS:-5}
local_ip=${PLAIT_LOCAL_IP:-11.166.42.141}
remote_ip=${PLAIT_REMOTE_IP:-215.193.188.234}
port=${PLAIT_TCP_PORT:-31300}
remote=${PLAIT_REMOTE:-pro5000}
remote_dir=${PLAIT_REMOTE_DIR:-/root/plait-src}
/opt/cuda129/bin/nvcc -O3 -std=c++17 scripts/tcp-twolevel.cu -o target/tcp-twolevel
tar -czf /tmp/tcp-twolevel-src.tgz scripts/tcp-twolevel.cu
ssh "$remote" "mkdir -p '$remote_dir' && tar -xzf - -C '$remote_dir'" < /tmp/tcp-twolevel-src.tgz
ssh "$remote" "cd '$remote_dir' && /usr/local/cuda/bin/nvcc -O3 -std=c++17 scripts/tcp-twolevel.cu -o target/tcp-twolevel"
rm -f /tmp/tcp-twolevel-local-*.log /tmp/tcp-twolevel-remote.log
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank target/tcp-twolevel "$rank" 12 0 "$bytes" "$iters" "$local_ip" "$remote_ip" "$port" >/tmp/tcp-twolevel-local-$rank.log 2>&1 &
done
ssh "$remote" "cd '$remote_dir' && for i in 0 1 2 3 4 5 6 7; do rank=\$((4+i)); CUDA_VISIBLE_DEVICES=\$i target/tcp-twolevel \"\$rank\" 12 0 '$bytes' '$iters' '$local_ip' '$remote_ip' '$port' > /tmp/tcp-twolevel-remote-\$rank.log 2>&1 & done; wait; cat /tmp/tcp-twolevel-remote-*.log" > /tmp/tcp-twolevel-remote.log 2>&1 &
wait || true
cat /tmp/tcp-twolevel-local-*.log
cat /tmp/tcp-twolevel-remote.log
