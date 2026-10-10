Follow-up with a GPU reproduction. Setup: A100-40GB, vLLM 0.20.0, torch 2.11.0+cu130, LMCache `dev@34d8027` built from source, Qwen2.5-1.5B-Instruct, `LMCACHE_BLEND_SPECIAL_STR=" # # "`, layerwise blending. The prompt is `system SEP question SEP document(~3.1k tokens) SEP suffix`.

I found **two independent causes**:

**1. Empty segments stop lookup, even on an exact repeat.**
- Env parsing strips `" # # "` to `"# #"`.
- `SegmentTokenDatabase` then drops the first token (`encode(...)[1:]`, the BOS assumption from item 3 above). For Qwen this leaves the single token `" #"` as the separator.
- The separator in the prompt contains `" #"` twice, so `process_tokens` yields an empty segment such as `[13,13)`.
- `store_layer` skips empty ranges. `lookup` doesn't: it hashes the empty segment, misses, and stops.

Per-segment debug logging from lookup, on an exact repeat of a stored prompt:
```
lookup seg [0,12)  hit_layers=28/28
lookup seg [13,13) hit_layers=0/28   <- empty segment, never stored
LMCache hit tokens: 12 (of 3161)
```

**2. Prefix-only hit counting.** This is the cause discussed above: `lookup` stops at the first missed segment, and `blend()` only sees `tokens[:lmcache_cached_tokens]`.

With a one-line fix for cause 1 (skip empty segments in `SegmentTokenDatabase.process_tokens`):

| Turn | `dev` | + fix 1 |
|---|---|---|
| new question, same document | 12 / 3161 | 12 / 3161 (cause 2) |
| exact repeat | 12 / 3161 | **3161 / 3161** |
| document moved before the question | 12 / 3161 | **3161 / 3161** |

So non-prefix reuse works once all segments are cached. (#3228's separator tokenization fix would also remove this particular trigger on Qwen. Skipping empty segments covers any adjacent separators, so the two complement each other.) I'll open a small PR for fix 1 with a unit test. Cause 2 needs a design decision because of the V1 "first N tokens computed" contract. I'm happy to work on it once there's agreement on an approach: non-prefix hit reporting in lookup, masked retrieval, and forced recompute of missed segments in the blender.

Also: on vLLM 0.31.0, CacheBlend fails before reaching this code, in every supported configuration:
- FlashAttention: the layerwise store raises `IndexError: Dimension out of range` (4-D per-layer KV).
- FlashInfer without sparse mode: rejected as unsupported.
- FlashInfer with sparse mode: `ImportError` for `flashinfer.page.block_sparse_indices_to_vector_sparse_offsets`.

I can file that separately if useful.
