#!/usr/bin/env bash
# Trimmed run on an already-set-up session (see run_realrag.sh for setup).
set -x
PY=/usr/bin/python3
export LIMIT=100
pkill -f run_realrag.sh; pkill -f real_rag.py
kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
LMC=/content/LMCache/lmcache/v1
variant() {
  case $1 in
    dev)   cp /content/td_dev.py $LMC/token_database.py; cp /content/bl_dev.py $LMC/compute/blend/blender.py ;;
    e5584) cp /content/td_fix.py $LMC/token_database.py; cp /content/bl_dev.py $LMC/compute/blend/blender.py ;;
    both)  cp /content/td_fix.py $LMC/token_database.py; cp /content/bl_fix.py $LMC/compute/blend/blender.py ;;
  esac
}
run() {  # dataset mode [variant]
  kill_gpu
  if [ -n "$3" ]; then variant $3; tag=$3; else tag=$2; fi
  $PY /content/real_rag.py $2 /content/data/$1.jsonl /content/rr_${1}_$tag.jsonl > /content/rrlog_${1}_$tag.txt 2>&1
  echo "EXIT ${1}_$tag $?"
}
run 2wikimqa recompute
run 2wikimqa lmcache both
run 2wikimqa lmcache e5584
run 2wikimqa lmcache dev
run musique recompute
run musique lmcache both
echo RR_DONE
