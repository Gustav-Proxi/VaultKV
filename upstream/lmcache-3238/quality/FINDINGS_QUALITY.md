# CacheBlend output-quality investigation (2026-10-10)

**Setup:**
- Colab A100-40GB, vLLM 0.20.0, LMCache at PR #5584 (so blend hits occur).
- Layerwise blending, check layer 1, ratio 0.15.
- Harness: `qrepro.py`, 32-token greedy outputs compared with full recompute on an exact repeat and on
  reordered docs. Debug hooks: `patch_debug*.py` (env-gated).

## Root cause 1: separator ("gap") KV is zero-filled (fixed)

- `VLLMBufferLayerwiseGPUConnector.batched_to_gpu` zero-fills positions that belong to no segment, i.e.
  separator tokens (`compute_gpu_buffer_obj.tensor[:, gap_positions] = 0.0`).
- The blender never recomputes those positions:
  - layers before the check layer write nothing back;
  - after the check layer, only the top-k deviating tokens are recomputed.
- Zero K/V at delimiter tokens corrupts the context at every recompute ratio.

**Evidence:**

| Variant | 1.5B repeat, matching prefix chars (median) | 7B artifacts repeat / reorder |
|---|---|---|
| blend (`dev` + #5584) | 5 | 8/8 / 7/8 |
| blend, no write-back (pure loaded KV) | 4 | – |
| LMCache without blending (control) | output = recompute 2/3 | – |
| **+ gap recompute** | **127** (= recompute 2/3) | **0/8 / 0/8** |

- The no-write-back ablation is still garbled. The non-blend layerwise control is correct, and all 28
  layers are flushed. Together these localize the bug to the blend-only loader steps: the fused RoPE
  passes its self-check, which leaves gap zeroing.
- Fix: in `LMCBlender.process_qkv`, copy fresh K/V rows for gap positions while all tokens are computed,
  and force gap positions into the recompute set at the check layer.
- TTFT cost: none (7B, 12k: 284 → 281 ms reordered).

## Root cause 2: flash-attn causal mask on scattered queries (reported, not fixed)

- After the check layer, `LMCFlashAttnBackend.forward_contiguous` calls `flash_attn_varlen_func` with the
  selected k queries, all N keys and `causal=True`.
- Flash-attn aligns the causal mask bottom-right when `seqlen_q != seqlen_k`. The i-th selected query
  (true position p_i) can therefore attend to keys up to `N - k + i ≥ p_i`, i.e. future tokens.

**Evidence:**
- A correct explicit-mask SDPA path (`LMC_FIX_CAUSAL=1`) improves agreement with recompute on reordered
  prompts:
  - 7B reorder median matching prefix: 2 → 36 chars.
  - 1.5B reorder answers now name the same doc as recompute in 3/3 trials.
- The dense SDPA is about 4× slower (1238 ms), so it is a probe, not a fix.
- Efficient fix idea: treat each selected query as its own length-1 sequence over a shared paged key
  buffer (block_table) with `seqused_k = p_i + 1`, which is how decode attention already works.

## Not discriminative

- Ground-truth task accuracy: the synthetic "largest figure" task is too hard. Even full recompute
  scores 2/8 and 0/8.
- Exact 32-token text match on 7B: it cascades on tiny numeric differences.
- So the artifact rate and the matching-prefix length are the reported metrics.

Raw data: `results/`.
