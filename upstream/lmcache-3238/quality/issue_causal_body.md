**Describe the bug**

After the check layer, `LMCBlender.process_qkv` keeps only the top-k selected tokens as queries, while keys/values cover all N tokens. `LMCFlashAttnBackend.forward_contiguous` then calls `flash_attn_varlen_func` with `cu_seqlens_q=[0, k]`, `cu_seqlens_k=[0, N]` and `causal=True`.

When `seqlen_q != seqlen_k`, flash-attn aligns the causal mask **bottom-right**: query i may attend to keys `j <= i + (N - k)`. The selected queries are scattered through the prompt (true positions `p_0 < ... < p_{k-1}`), not the last k tokens. So the i-th selected query can see keys up to `N - k + i`, which is past its true position `p_i`. **Recomputed tokens attend to future tokens**, and their recomputed K/V (written into the paged cache) are wrong.

**Evidence** (A100, vLLM 0.20.0, layerwise blending, check layer 1, ratio 0.15, with #5584 and the separator fix applied)

I replaced the attention for selected queries with an explicit mask on true positions (`mask[i, j] = j <= p_i`, via `scaled_dot_product_attention`). Comparing 32-token greedy outputs with full recompute on reordered documents:

| | Qwen2.5-7B, 12k tokens: matching prefix (median chars) | Qwen2.5-1.5B: answer names the same doc as recompute |
|---|---|---|
| current flash-attn path | 2 | 0/3 |
| explicit true-causal mask | 36 | 3/3 |

On exact repeats the effect is smaller (7B: 50 → 62 chars).

**Proposed fix**

The dense SDPA mask is about 4× slower (7B, 12k: 281 ms → 1238 ms), so it's only a correctness probe. An efficient exact option: treat each selected query as its own length-1 sequence over a shared paged key buffer. Use `block_table` rows that all point at the same blocks, with `seqused_k = p_i + 1` and `cu_seqlens_q = arange(k + 1)`. That's the same pattern decode attention already uses, so cost stays close to the current path.

Happy to implement this if the approach sounds right.
