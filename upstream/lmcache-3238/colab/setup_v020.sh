set -ex
PY=/usr/bin/python3
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
uv pip install --python $PY -q "vllm==0.20.0" "torch==2.11.0" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
$PY -c "import vllm, torch; print('VLLM', vllm.__version__, 'TORCH', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
cd /content/LMCache && rm -rf build && find lmcache -name "*.so" -delete
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=12
uv pip install --python $PY --no-build-isolation -e . 2>&1 | tail -5
$PY -c "from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl; print('LMCACHE_OK')" 2>&1 | tail -1
f=$($PY -c "import vllm.v1.worker.gpu_worker as g; print(g.__file__)")
grep -n "ensure_kv_transfer_initialized\|def load_model\|model_runner.load_model\|def init_worker_distributed_environment" $f
echo SETUP_DONE
