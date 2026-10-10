#!/usr/bin/env bash
# Controls (LMCache without blending) + flush counting for the blend path.
set -x
PY=/usr/bin/python3
$PY /content/patch_debug2.py
kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
T=q2
kill_gpu; NOBLEND=1 LAYERWISE=True $PY /content/qrepro.py lmcache /content/${T}_prefix_lw.jsonl > /content/${T}_log_prefix_lw.txt 2>&1; echo "EXIT prefix_lw $?"
kill_gpu; NOBLEND=1 LAYERWISE=False $PY /content/qrepro.py lmcache /content/${T}_prefix_nolw.jsonl > /content/${T}_log_prefix_nolw.txt 2>&1; echo "EXIT prefix_nolw $?"
kill_gpu; LMC_DBG_NOWRITE=1 $PY /content/qrepro.py lmcache /content/${T}_nowrite.jsonl > /content/${T}_log_nowrite.txt 2>&1; echo "EXIT nowrite $?"
echo Q2_DONE
