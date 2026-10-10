#!/usr/bin/env bash
# 7B confirmation: quality (with ground truth) + TTFT for baseline vs gap fix vs gap+causal.
set -x
PY=/usr/bin/python3
export MODEL=Qwen/Qwen2.5-7B-Instruct DOC_TOKENS=3000 N_DOCS=4 TRIALS=8
kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
T=q7
kill_gpu; $PY /content/qrepro.py recompute /content/${T}_recompute.jsonl > /content/${T}_log_recompute.txt 2>&1; echo "EXIT recompute $?"
kill_gpu; $PY /content/qrepro.py lmcache /content/${T}_blend.jsonl > /content/${T}_log_blend.txt 2>&1; echo "EXIT blend $?"
kill_gpu; LMC_GAPFIX=1 $PY /content/qrepro.py lmcache /content/${T}_gap.jsonl > /content/${T}_log_gap.txt 2>&1; echo "EXIT gap $?"
kill_gpu; LMC_GAPFIX=1 LMC_FIX_CAUSAL=1 $PY /content/qrepro.py lmcache /content/${T}_gap_causal.jsonl > /content/${T}_log_gap_causal.txt 2>&1; echo "EXIT gap_causal $?"
# TTFT (bench_blend.py, 3 trials, single-step prefill)
export TRIALS=3 MAX_BATCHED=16384
kill_gpu; $PY /content/bench_blend.py lmcache /content/${T}_ttft_blend.jsonl > /content/${T}_tlog_blend.txt 2>&1; echo "EXIT ttft_blend $?"
kill_gpu; LMC_GAPFIX=1 $PY /content/bench_blend.py lmcache /content/${T}_ttft_gap.jsonl > /content/${T}_tlog_gap.txt 2>&1; echo "EXIT ttft_gap $?"
kill_gpu; LMC_GAPFIX=1 LMC_FIX_CAUSAL=1 $PY /content/bench_blend.py lmcache /content/${T}_ttft_gap_causal.jsonl > /content/${T}_tlog_gap_causal.txt 2>&1; echo "EXIT ttft_gap_causal $?"
echo Q4_DONE
