#!/usr/bin/env bash
set -x
PY=/usr/bin/python3
$PY /content/patch_debug3.py
kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
T=q3
kill_gpu; LMC_GAPFIX=1 LMC_DBG_NOWRITE=1 $PY /content/qrepro.py lmcache /content/${T}_gap_nowrite.jsonl > /content/${T}_log_gap_nowrite.txt 2>&1; echo "EXIT gap_nowrite $?"
kill_gpu; LMC_GAPFIX=1 $PY /content/qrepro.py lmcache /content/${T}_gap.jsonl > /content/${T}_log_gap.txt 2>&1; echo "EXIT gap $?"
kill_gpu; LMC_GAPFIX=1 LMC_FIX_CAUSAL=1 $PY /content/qrepro.py lmcache /content/${T}_gap_causal.jsonl > /content/${T}_log_gap_causal.txt 2>&1; echo "EXIT gap_causal $?"
echo Q3_DONE
