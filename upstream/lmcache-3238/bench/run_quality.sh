#!/usr/bin/env bash
# Quality check: recompute vs lmcache_fix at recompute ratios 0.15 and 0.5 (single-step prefill).
set -x
PY=/usr/bin/python3
export MAX_BATCHED=16384 TRIALS=3
kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
cp /content/td_fix.py /content/LMCache/lmcache/v1/token_database.py
kill_gpu; $PY /content/bench_blend.py recompute /content/q_recompute.jsonl > /content/qlog_recompute.txt 2>&1; echo "EXIT recompute $?"
for r in 0.15 0.5; do
  kill_gpu; RATIO=$r $PY /content/bench_blend.py lmcache /content/q_fix_$r.jsonl > /content/qlog_fix_$r.txt 2>&1; echo "EXIT fix_$r $?"
done
echo Q_DONE
