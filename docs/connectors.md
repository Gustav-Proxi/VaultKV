# Experimental vLLM / LMCache connector

The first connector implements **guarded prefix reuse**, with a dynamic vLLM V1
plugin, a trusted in-process generation adapter, and SQLite transfer permits.
It has CPU lifecycle/race tests. **GPU execution and performance are unverified**:
this development machine is macOS ARM64 without NVIDIA CUDA.

## Pinned target

| Component | Version / immutable revision |
| --- | --- |
| vLLM | `0.12.0` / `4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e` |
| LMCache | `0.3.9` / `0b4c03720a3e04accfcb2283fc3c22c55581776a` |
| Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

These are source-inspected integration targets, not a claim of latest releases or
an established GPU compatibility matrix. The plugin rejects other package versions.
LMCache also requires a compatible torch/CUDA build: package version agreement
alone is insufficient. See [upstream compatibility guidance](https://docs.lmcache.ai/getting_started/compatibility.html).

The implementation follows the pinned [vLLM V1 base](https://github.com/vllm-project/vllm/blob/4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e/vllm/distributed/kv_transfer/kv_connector/v1/base.py),
[LMCache dynamic connector](https://github.com/LMCache/LMCache/blob/0b4c03720a3e04accfcb2283fc3c22c55581776a/lmcache/integration/vllm/lmcache_connector_v1.py),
and [request-tag cache key implementation](https://github.com/LMCache/LMCache/blob/0b4c03720a3e04accfcb2283fc3c22c55581776a/lmcache/utils.py).

## Local diagnostics

```sh
uv run vaultkv connector-doctor
uv run vaultkv connector-config --db data/local-demo.sqlite3
uv run pytest tests/connectors
```

Diagnostics never import torch or initialize CUDA. The dashboard's Caches & epochs
view shows the target versions, missing prerequisites and recorded worker transfers.
Lookup predictions and actual loaded-token counts are distinct events. Tensor bytes
and hit rate remain unknown. Recorded transfers are history, not a live-engine heartbeat.

## Linux CUDA setup and smoke test

Use a separate Python 3.12 environment on a CUDA development host with a matching
compiler/toolkit and enough GPU memory for Qwen. Do not install this stack into
the ordinary macOS environment. LMCache 0.3.9 wheels were built against torch 2.8;
build against the torch installed by the pinned vLLM to avoid mixing native ABIs.

```sh
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/bin/python vllm==0.12.0
uv pip install --python .venv-gpu/bin/python ninja 'packaging>=24.2' \
  'setuptools>=77.0.3,<81' 'setuptools_scm>=8' wheel
uv pip install --python .venv-gpu/bin/python --no-build-isolation \
  --no-binary lmcache -r requirements/gpu.txt
uv pip install --python .venv-gpu/bin/python -e .
.venv-gpu/bin/python benchmarks/connector_gpu_smoke.py
```

The opt-in smoke test creates a temporary corpus, runs cold and warm real
inference, requires a nonzero observed warm transfer, checks the answer, then
revokes access after another generation and verifies zero released outputs.
It does not test restart persistence: this first configuration uses CPU RAM only.
Native compilation and this smoke test still need execution on the target host.

To query an ingested database on that host:

```sh
.venv-gpu/bin/vaultkv ask 'What is the hotel limit?' \
  --db data/local-demo.sqlite3 --backend vllm --retrieval bm25 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --model-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

Each CLI invocation starts a new engine. Use the smoke test or retain one
`VLLMClient` in Python to observe warm reuse across requests. The dashboard
composer still uses Ollama; vLLM requests launched through the CLI appear in its
history when both use the same database. Hybrid embedding retrieval is not yet
wired to this generation backend.

## Transfer and output contracts

1. A trusted producer registers a context with the full evidence read set and
   issues an opaque permit bound to the exact prompt token IDs.
2. The namespace includes workspace/user identity and user/policy/source
   epochs, content versions/hashes, context, prompt tokens and model profile.
   K itself is excluded so unchanged evidence can survive unrelated publication.
   Sharing is deliberately per-user and exact-prompt in this first version.
3. Scheduler lookup and allocation recheck current authorization. Metadata carries
   the permit and the namespace through LMCache's request configuration mechanism.
4. Worker loads and saves validate the permit and prompt prefix under the SQLite
   mutation lock. Device synchronization completes before releasing that lock,
   including error paths. Short loads fail; denied loads never become recomputation.
5. Generation runs outside that lock. The existing TxnRAG release gate authorizes
   the buffered answer and citations again before delivery.

All participating processes must use the same absolute database path on one host.
The database, producer, engine and model runtime remain trusted. This does not
prevent privileged code from bypassing the API, attest model weights, erase old
RAM entries, or recall already delivered output.

## Deliberate limits

Single GPU, one sequence, eager execution, fp16, synchronous transfers, local CPU
storage, and prefix reuse only. vLLM APC, chunked prefill, asynchronous/layerwise
loading, blending, disaggregation and remote storage are disabled. A revoked or
malformed worker batch raises and can terminate the engine; automatic per-request
abort/recovery is not implemented. Restart the experimental engine after such an
error. Outstanding permits become unusable when their TxnRAG request closes;
a process crash can leave trusted internal rows requiring later cleanup.

Non-prefix blending/LMCache issue #3238, multi-GPU behavior, distributed metadata,
streaming, scope-equivalence sharing, timing-attack protection, throughput and
TTFT claims remain future work. The CPU tests use explicit transfer fixtures;
they are not tensor or GPU benchmark results.
