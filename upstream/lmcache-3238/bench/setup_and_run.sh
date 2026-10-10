#!/usr/bin/env bash
# Colab A100: vLLM 0.20.0 + LMCache built from the PR branch, then all benchmark modes.
# Usage: nohup bash setup_and_run.sh > /content/bench.log 2>&1 &
set -x
PY=/usr/bin/python3
cd /content

uv pip install --python $PY -q "vllm==0.20.0" "torch==2.11.0" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
$PY -c "import vllm, torch; print('VERSIONS vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"

if [ ! -d LMCache ]; then
  git clone -q https://github.com/LMCache/LMCache.git
fi
cd LMCache
git fetch -q https://github.com/Gustav-Proxi/LMCache.git fix/segment-db-empty-segments
git checkout -q FETCH_HEAD
git log -1 --format='LMCACHE_HEAD %h %s'
# dev = the PR's parent; fix = the PR head. Only token_database.py differs.
git show HEAD~1:lmcache/v1/token_database.py > /content/td_dev.py
git show HEAD:lmcache/v1/token_database.py > /content/td_fix.py
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=12
uv pip install --python $PY -q -r requirements/build.txt
uv pip install --python $PY --no-build-isolation -e . 2>&1 | tail -3
cd /content
$PY /content/patch_vllm.py

kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }

for mode in recompute apc; do
  kill_gpu
  $PY /content/bench_blend.py $mode /content/res_$mode.jsonl > /content/log_$mode.txt 2>&1
  echo "EXIT $mode $?"
done
for variant in dev fix; do
  kill_gpu
  cp /content/td_$variant.py /content/LMCache/lmcache/v1/token_database.py
  $PY /content/bench_blend.py lmcache /content/res_lmcache_$variant.jsonl > /content/log_lmcache_$variant.txt 2>&1
  echo "EXIT lmcache_$variant $?"
done
cp /content/td_fix.py /content/LMCache/lmcache/v1/token_database.py
echo BENCH_DONE
