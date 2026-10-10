#!/usr/bin/env bash
# Rerun all modes on an already-set-up session. Usage: TAG=... MAX_BATCHED=... bash run_modes.sh
set -x
PY=/usr/bin/python3
kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
for mode in recompute apc; do
  kill_gpu
  $PY /content/bench_blend.py $mode /content/res_${TAG}_$mode.jsonl > /content/log_${TAG}_$mode.txt 2>&1
  echo "EXIT $mode $?"
done
for variant in dev fix; do
  kill_gpu
  cp /content/td_$variant.py /content/LMCache/lmcache/v1/token_database.py
  $PY /content/bench_blend.py lmcache /content/res_${TAG}_lmcache_$variant.jsonl > /content/log_${TAG}_lmcache_$variant.txt 2>&1
  echo "EXIT lmcache_$variant $?"
done
cp /content/td_fix.py /content/LMCache/lmcache/v1/token_database.py
echo RUN_DONE
