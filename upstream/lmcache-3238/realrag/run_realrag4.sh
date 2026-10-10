#!/usr/bin/env bash
set -x
PY=/usr/bin/python3
kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
LMC=/content/LMCache/lmcache/v1
cp /content/td_fix.py $LMC/token_database.py; cp /content/bl_fix.py $LMC/compute/blend/blender.py
export WARM_STYLE=sep
# LMCache leaks lookup pins here (stalls until the 300 s pin monitor fires); force-release
# them fast. Safe: the CPU cache is far from full, so nothing is evicted while unpinned.
export LMCACHE_PIN_TIMEOUT_SEC=5 LMCACHE_PIN_CHECK_INTERVAL_SEC=1
pkill -f run_realrag3.sh; pkill -f real_rag.py
kill_gpu; LIMIT=100 $PY /content/real_rag.py lmcache /content/data/2wikimqa.jsonl /content/rr4_2wikimqa_both.jsonl > /content/rr4log_2wikimqa_both.txt 2>&1; echo "EXIT 2wiki_both $?"
export MAX_PROMPT=15500 LIMIT=50
kill_gpu; $PY /content/real_rag.py recompute /content/data/musique.jsonl /content/rr4_musique_recompute.jsonl > /content/rr4log_musique_recompute.txt 2>&1; echo "EXIT musique_recompute $?"
kill_gpu; $PY /content/real_rag.py lmcache /content/data/musique.jsonl /content/rr4_musique_both.jsonl > /content/rr4log_musique_both.txt 2>&1; echo "EXIT musique_both $?"
echo RR3_DONE
