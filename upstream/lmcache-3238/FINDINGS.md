# LMCache #3238 findings (2026-10-09)

Setup: Colab A100-40GB, vLLM 0.20.0, torch 2.11.0+cu130, LMCache `dev@34d8027` built from source,
Qwen2.5-1.5B-Instruct, `LMCACHE_BLEND_SPECIAL_STR=" # # "`, layerwise blending, recompute ratio 0.15.
Workload: `repro_3238_gpu.py` (system | question | ~3.1k-token document | suffix, separated by SEP).

## Two independent causes

1. **Empty segments: lookup stops even on an exact repeat.**
   - `LMCACHE_BLEND_SPECIAL_STR=" # # "` is whitespace-stripped by env parsing (`config_base.py`
     unquote → `"# #"`). `SegmentTokenDatabase` then drops the first token (`encode(...)[1:]`, a BOS
     assumption that Qwen breaks), leaving the 1-token separator `" #"`.
   - The prompt separator `" # # "` contains `" #"` twice, so `process_tokens` yields an empty
     segment, e.g. `[13,13)`.
   - `store_layer` skips empty ranges. `lookup` hashes the empty segment, finds no key and stops.
   - Fix candidate: `process_tokens` must not yield empty segments (`colab/fix_empty.py`).

2. **Prefix-only hit counting: a changed segment hides everything after it.**
   - `LMCacheEngine.lookup` returns the end of the leading run of hit segments. The worker blends
     only `tokens[:lmcache_cached_tokens]`.
   - A changed question therefore hides the cached document behind it.
   - Fixing this needs non-prefix hit reporting, masked retrieval and forced recompute of missed
     segments. It interacts with vLLM V1's "first N tokens computed" contract, so the approach
     needs maintainer input.

## Measured (LMCache hit tokens / prompt tokens)

| Turn | dev@34d8027 | + fix 1 |
|---|---|---|
| turn2: new question, same doc | 12 / 3161 | 12 / 3161 (cause 2) |
| turn3: exact repeat | 12 / 3161 | **3161 / 3161** |
| turn4: doc moved before question | 12 / 3161 | **3161 / 3161** (non-prefix reuse) |

Logs: `logs/run_lmcache.log` (before), `logs/run_dbg.log` (segment debug), `logs/run_fix1.log` (after).

## Caveats

- **TTFT is not meaningful yet.** On a 1.5B model with ~3k tokens a cold prefill takes about 70 ms.
  Blended turns took 137–573 ms (the first blend call warms up). Benchmarks need a 7B/8B model and
  8–16k contexts, where prefill dominates.
- **Current vLLM can't run CacheBlend at all.** With vLLM 0.31.0, every supported blend
  configuration fails before #3238 matters:
  - FlashAttention: the layerwise store hits `IndexError` on the 4-D per-layer KV tensor.
  - FlashInfer without sparse mode: rejected as unsupported.
  - FlashInfer with sparse mode: imports `flashinfer.page.block_sparse_indices_to_vector_sparse_offsets`,
    which no longer exists.
- **The vLLM 0.20 worker patch is needed for blending.** It registers the model in `load_model`
  (`colab/patch_vllm.py`). The README patch for older versions doesn't apply.
