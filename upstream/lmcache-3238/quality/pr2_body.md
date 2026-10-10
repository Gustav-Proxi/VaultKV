**What this PR does / why we need it**:

Blended outputs are corrupted, even on an **exact repeat** of a cached prompt: repeated words such as "the the", stray commas, and wrong figures. Raising the recompute ratio doesn't help.

Root cause: `VLLMBufferLayerwiseGPUConnector.batched_to_gpu` zero-fills positions that belong to no cached segment, i.e. the separator tokens (`compute_gpu_buffer_obj.tensor[:, gap_positions] = 0.0`). `LMCBlender.process_qkv` never recomputes them:
- layers before the check layer write nothing back;
- after the check layer, only the top-k deviating tokens are recomputed.

So every layer keeps K = V = 0 at the delimiter positions, and that corrupts the context.

This PR recomputes the gap positions:
- before the check layer (all tokens are computed), it copies the fresh K/V rows for gap positions into the loaded buffer;
- at the check layer, it forces gap positions into the recompute set (`torch.unique` keeps indices sorted).

How the cause was isolated (A100, vLLM 0.20.0, with #5584 applied so blend hits happen):
- **No write-back ablation:** blending but never writing recomputed K/V back, i.e. pure loaded KV. Still garbled.
- **LMCache without blending** (layerwise and non-layerwise): output matches full recompute, so KV transfer and layout are fine.
- **All 28 layers are flushed per load**, and the fused RoPE passes its self-check. That leaves the gap zero-fill.
- **Recomputing only the gap positions** (with write-back otherwise disabled) restores outputs.

Results, comparing 32-token greedy outputs with full recompute:

| | Qwen2.5-1.5B, exact repeat: matching prefix (median chars) | Qwen2.5-7B, 12k tokens: outputs with artifacts (repeat / reordered docs) |
|---|---|---|
| before | 5 | 8/8 / 7/8 |
| after | 127 (same as LMCache without blending) | 0/8 / 0/8 |

TTFT is unchanged (7B, 12k tokens, reordered docs: 284 ms → 281 ms). Only a handful of separator positions are added to the recompute set.

Refs #3238. Complements #5584, which lets blend lookups hit in the first place.

**Special notes for your reviewers**:

- The new tests drive `process_qkv` with a fake loaded buffer and identity RoPE, so they need no GPU or vLLM model. The two gap tests fail on `dev`; the no-gap case passes with and without this change.
- A separate issue remains in `LMCFlashAttnBackend.forward_contiguous`. It passes the selected queries with all keys and `causal=True`, and flash-attn aligns the causal mask bottom-right when `seqlen_q != seqlen_k`, so scattered selected queries can attend to future keys. Filed separately as #5596, with measurements. It's independent of this fix.

**If applicable**:

- [ ] this PR contains user facing changes - docs added
- [x] this PR contains unit tests
