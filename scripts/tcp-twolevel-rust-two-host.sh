#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bytes=${PLAIT_TCP_BYTES:-1048576}
iters=${PLAIT_TCP_ITERS:-5}
local_ip=${PLAIT_LOCAL_IP:-11.166.42.141}
remote_ip=${PLAIT_REMOTE_IP:-215.193.188.234}
port=${PLAIT_TCP_PORT:-32700}
remote=${PLAIT_REMOTE:-pro5000}
remote_dir=${PLAIT_REMOTE_DIR:-/root/plait-src}

cargo build --release -q -p plait-collective --bin tcp-twolevel
tar --exclude=target --exclude=.git -czf /tmp/plait-rust-src.tgz .
ssh "$remote" "rm -rf '$remote_dir' && mkdir -p '$remote_dir' && tar -xzf - -C '$remote_dir'" < /tmp/plait-rust-src.tgz
ssh "$remote" "cd '$remote_dir' && export PATH=/root/.cargo/bin:\$PATH && cargo build --release -q -p plait-collective --bin tcp-twolevel"

rm -f /dev/shm/plait-rust-hybrid-$port /tmp/tcp-rust-local-*.log /tmp/tcp-rust-remote.log
ssh "$remote" "rm -f /dev/shm/plait-rust-hybrid-$port"
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank target/release/tcp-twolevel "$rank" 12 0 "$bytes" "$iters" "$local_ip" "$remote_ip" "$port" >/tmp/tcp-rust-local-$rank.log 2>&1 &
done
ssh "$remote" "cd '$remote_dir' && export PATH=/root/.cargo/bin:\$PATH; for i in 0 1 2 3 4 5 6 7; do rank=\$((4+i)); CUDA_VISIBLE_DEVICES=\$i target/release/tcp-twolevel \"\$rank\" 12 0 '$bytes' '$iters' '$local_ip' '$remote_ip' '$port' > /tmp/tcp-rust-remote-\$rank.log 2>&1 & done; wait; cat /tmp/tcp-rust-remote-*.log" > /tmp/tcp-rust-remote.log 2>&1 &
wait || true
cat /tmp/tcp-rust-local-*.log
cat /tmp/tcp-rust-remote.log
