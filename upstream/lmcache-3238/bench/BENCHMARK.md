# CacheBlend non-prefix reuse benchmark (LMCache PR #5584)

**Setup:**
- Colab A100-SXM4-40GB, vLLM 0.20.0, torch 2.11.0+cu130, LMCache built at PR #5584
  (`dc24de1`, parent = `dev`). "dev" vs "fix" swaps only `lmcache/v1/token_database.py`.
- Qwen2.5-7B-Instruct, bf16, `enforce_eager`, `max_num_batched_tokens=16384`.
- Blending: layerwise, check layer 1, recompute ratio 0.15, CPU backend. Prefix caching is off for the
  LMCache runs.

**Workload** (`bench_blend.py`): `system | D1 | D2 | D3 | D4 | question | "Answer:"` with `" # # "`
separators, 4 synthetic docs of 3,000 tokens each, 12,048 prompt tokens in total.
- Per trial, with fresh docs: a cold request, then the same docs **reordered**, then an **exact repeat**.
- TTFT = wall time of a 1-token greedy generation.
- 1 warmup trial, then 5 measured trials. Values are medians.

## Speed (12,048-token prompt)

| Config | cold | reordered docs | exact repeat | LMCache hit tokens (reorder) |
|---|---|---|---|---|
| full recompute | 815 ms | 817 ms | 817 ms | – |
| vLLM APC | 811 ms | 812 ms | **36 ms** | – |
| LMCache `dev` | 919 ms | 916 ms | 916 ms | 13 / 12,048 |
| LMCache + #5584 | 917 ms | **282 ms** | 282 ms | **12,048 / 12,048** |

- Reordered docs: **3.2× faster than `dev`** and 2.9× faster than vLLM APC, which can't reuse
  reordered context. The spread was tight: min 281 ms, max 285 ms.
- Exact repeat: APC is faster still (36 ms, pure prefix hit, no recompute). Blending's value is the
  non-prefix case.
- Raw data: `results_b16k/`.

## Output quality: blend path diverges even on exact repeats

We compared 32-token greedy outputs with full recompute (`results_quality/`, 3 trials):

| Config | exact repeat identical | reorder identical |
|---|---|---|
| LMCache + #5584, ratio 0.15 | 0/3 | 0/3 |
| LMCache + #5584, ratio 0.50 | 0/3 | 0/3 |
| (reference) APC, reorder | – | 4/5 |

- Blended outputs are fluent but repeatedly show artifacts such as **"the the"** and wrong figures.
- An exact repeat puts every token at its stored position, so blended output should essentially match
  recompute. Raising the ratio to 0.5 does not fix it.
- This points to a correctness problem in the blend path, not CacheBlend's expected approximation
  error. `dev` never reached this path (no hits), so #5584 exposes it but does not cause it.
- Observations:
  - Separator tokens are never stored or retrieved: `Retrieved 12036 out of 12048` = 6 separators ×
    2 non-segment tokens.
  - On a full hit, vLLM also recomputes the last token while `blend()` processes
    `tokens[:lmcache_cached_tokens]`.
  - The root cause is unconfirmed.

## Chunked prefill splits segments

With vLLM's default `max_num_batched_tokens=8192`, the cold 12k prompt is stored in two steps
(`Stored 8186 out of 8192`, then `Stored 3020 out of 12048`). The segment that crosses the step
boundary is stored as two partial segments whose keys never match the whole segment. The next
lookup stops there, so reorder hit only 3,016–9,022 tokens (`results_chunked8192/`). The benchmark
above uses 16,384 to isolate #5584.

## vLLM 0.31.0

CacheBlend fails before reaching any of this, in every supported configuration:
- FlashAttention: the layerwise store raises `IndexError` (4-D per-layer KV).
- FlashInfer without sparse mode: rejected.
- FlashInfer with sparse mode: missing `flashinfer.page.block_sparse_indices_to_vector_sparse_offsets`.

## Reproduce

On a Colab A100:
1. `bash setup_and_run.sh` (installs vLLM 0.20.0, builds LMCache at the PR, runs all modes; needs
   `../colab/patch_vllm.py` in `/content`).
2. `TAG=b16k MAX_BATCHED=16384 bash run_modes.sh`
3. `bash run_quality.sh`
