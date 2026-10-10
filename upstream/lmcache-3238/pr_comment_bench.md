Benchmark on a larger model, as promised.

**Setup:** A100-40GB, vLLM 0.20.0, Qwen2.5-7B-Instruct.
- Prompt: `system | 4 × 3k-token docs | question`, 12,048 tokens, `" # # "` separators.
- Blending: layerwise, check layer 1, ratio 0.15, `max_num_batched_tokens=16384`.
- "dev" vs "this PR" swaps only `token_database.py`.
- 5 trials with fresh docs each; medians reported.

| Config | reordered docs | exact repeat | hit tokens (reorder) |
|---|---|---|---|
| full recompute | 817 ms | 817 ms | – |
| vLLM APC | 812 ms | 36 ms | – |
| LMCache `dev` | 916 ms | 916 ms | 13 / 12,048 |
| LMCache + this PR | **282 ms** | 282 ms | **12,048 / 12,048** |

With the docs reordered, that's **3.2× faster than `dev`** and 2.9× faster than APC.

Three more things I found that reviewers should know. None of them is caused by this PR, but this PR is what lets the blend path run at all:

1. **Output quality diverges even on exact repeats.**
   - 32-token greedy outputs differed from full recompute in 3/3 exact-repeat trials and 3/3 reorder trials. They show artifacts like "the the" and wrong figures, and raising the ratio to 0.5 doesn't help.
   - On an exact repeat every token keeps its stored position, so this looks like a correctness bug in the blend path rather than the expected approximation error.
   - Possibly relevant: separator tokens are never stored or retrieved (`Retrieved 12036 out of 12048` = 6 separators × 2 tokens). Also, on a full hit vLLM recomputes the last token while `blend()` processes `tokens[:lmcache_cached_tokens]`. I haven't confirmed the root cause.
2. **Chunked prefill splits segments.** With the default `max_num_batched_tokens=8192`, a 12k prompt is stored in two steps. The segment that crosses the boundary is stored as two partial segments that never match the whole segment, so reorder hit only 3,016–9,022 tokens.
3. **vLLM 0.31.0:** CacheBlend fails in every supported configuration, as listed in #3238.

I'm happy to file 1 and 2 as separate issues with repro scripts. Full write-up, raw JSONL results and Colab scripts are ready to share as a gist if that helps.
