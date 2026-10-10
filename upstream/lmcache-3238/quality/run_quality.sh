#!/usr/bin/env bash
# Colab A100: setup (vLLM 0.20 + LMCache at PR #5584) + debug hooks + all variants.
# Usage: MODEL=... nohup bash run_quality.sh > /content/quality.log 2>&1 &
set -x
PY=/usr/bin/python3
cd /content
if ! $PY -c "import vllm" 2>/dev/null || [ "$($PY -c 'import vllm;print(vllm.__version__)')" != "0.20.0" ]; then
  uv pip install --python $PY -q "vllm==0.20.0" "torch==2.11.0" \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
fi
if [ ! -d LMCache ]; then
  git clone -q https://github.com/LMCache/LMCache.git
  cd LMCache
  git fetch -q https://github.com/Gustav-Proxi/LMCache.git fix/segment-db-empty-segments
  git checkout -q FETCH_HEAD
  export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=12
  uv pip install --python $PY -q -r requirements/build.txt
  uv pip install --python $PY --no-build-isolation -e . 2>&1 | tail -3
  cd /content
fi
$PY /content/patch_vllm.py
$PY /content/patch_debug.py
echo SETUP_DONE

kill_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done; sleep 3; }
TAG=${TAG:-q}
kill_gpu; $PY /content/qrepro.py recompute /content/${TAG}_recompute.jsonl > /content/${TAG}_log_recompute.txt 2>&1; echo "EXIT recompute $?"
kill_gpu; LMC_DBG_LOG=1 $PY /content/qrepro.py lmcache /content/${TAG}_blend.jsonl > /content/${TAG}_log_blend.txt 2>&1; echo "EXIT blend $?"
kill_gpu; LMC_DBG_NOWRITE=1 $PY /content/qrepro.py lmcache /content/${TAG}_nowrite.jsonl > /content/${TAG}_log_nowrite.txt 2>&1; echo "EXIT nowrite $?"
kill_gpu; LMC_DBG_LOG=1 LMC_FIX_CAUSAL=1 $PY /content/qrepro.py lmcache /content/${TAG}_fix.jsonl > /content/${TAG}_log_fix.txt 2>&1; echo "EXIT fix $?"
echo Q_DONE
