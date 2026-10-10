#!/usr/bin/env bash
# Colab A100: real-RAG benchmark across recompute and three LMCache variants.
# LMCache source = PR #5595 branch; variants swap token_database.py / blender.py:
#   dev    : neither fix
#   e5584  : empty-segment fix only (blend hits, separator KV still zero)
#   both   : #5584 + #5595
set -x
PY=/usr/bin/python3
cd /content
uv pip install --python $PY -q "vllm==0.20.0" "torch==2.11.0" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
if [ ! -d LMCache ]; then
  git clone -q https://github.com/LMCache/LMCache.git
fi
cd LMCache
git fetch -q https://github.com/Gustav-Proxi/LMCache.git fix/blend-recompute-separator-gaps
git checkout -q FETCH_HEAD
git fetch -q https://github.com/Gustav-Proxi/LMCache.git fix/segment-db-empty-segments
git show FETCH_HEAD:lmcache/v1/token_database.py > /content/td_fix.py
git show HEAD:lmcache/v1/token_database.py > /content/td_dev.py
git show HEAD:lmcache/v1/compute/blend/blender.py > /content/bl_fix.py
git show HEAD~1:lmcache/v1/compute/blend/blender.py > /content/bl_dev.py
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=12
uv pip install --python $PY -q -r requirements/build.txt
uv pip install --python $PY --no-build-isolation -e . 2>&1 | tail -3
cd /content
$PY /content/patch_vllm.py

mkdir -p /content/data
if [ ! -f /content/data/2wikimqa.jsonl ]; then
  wget -q https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip -O /content/lb.zip
  unzip -q -o /content/lb.zip -d /content/lbz
  find /content/lbz -name "2wikimqa.jsonl" -exec cp {} /content/data/ \;
  find /content/lbz -name "musique.jsonl" -exec cp {} /content/data/ \;
fi
ls -la /content/data
echo SETUP_DONE

kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
LMC=/content/LMCache/lmcache/v1
for ds in 2wikimqa musique; do
  kill_gpu; $PY /content/real_rag.py recompute /content/data/$ds.jsonl /content/rr_${ds}_recompute.jsonl > /content/rrlog_${ds}_recompute.txt 2>&1; echo "EXIT ${ds}_recompute $?"
  for v in dev e5584 both; do
    case $v in
      dev)   cp /content/td_dev.py $LMC/token_database.py; cp /content/bl_dev.py $LMC/compute/blend/blender.py ;;
      e5584) cp /content/td_fix.py $LMC/token_database.py; cp /content/bl_dev.py $LMC/compute/blend/blender.py ;;
      both)  cp /content/td_fix.py $LMC/token_database.py; cp /content/bl_fix.py $LMC/compute/blend/blender.py ;;
    esac
    kill_gpu; $PY /content/real_rag.py lmcache /content/data/$ds.jsonl /content/rr_${ds}_$v.jsonl > /content/rrlog_${ds}_$v.txt 2>&1; echo "EXIT ${ds}_$v $?"
  done
done
echo RR_DONE
