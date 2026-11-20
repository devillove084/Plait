#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q -p plait-node
bin=target/debug/plait-node
manifest=$(mktemp)
trap 'kill "$b_pid" 2>/dev/null || true; rm -f "$manifest"' EXIT
for i in $(seq 1 12); do
  echo "$("$bin" id)@127.0.0.1:$((7300 + i))" >>"$manifest"
done
"$bin" matrix --manifest "$manifest" --start 4 --count 8 --hold 40 --asymmetric >/tmp/plait-asym-b.log 2>&1 &
b_pid=$!
sleep 2
"$bin" matrix --manifest "$manifest" --start 0 --count 4 --driver --hold 20 --asymmetric
