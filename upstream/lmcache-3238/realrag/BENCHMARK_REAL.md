# CacheBlend on real RAG data (LongBench 2WikiMQA), 2026-10-10

**Setup:**
- Colab A100-40GB, vLLM 0.20.0, Qwen2.5-7B-Instruct, `max_num_batched_tokens=16384`.
- Blending: layerwise, check layer 1, ratio 0.15.
- LMCache variants swap two files on the PR #5595 branch:
  - `dev`: neither fix
  - `#5584 only`: empty-segment fix
  - `both`: #5584 + #5595
- Protocol (CacheBlend's evaluation):
  - Each question's ~10 passages are first sent alone (precomputed chunk KV).
  - Then `instruction | P1..Pn | question | "Answer:"` is answered.
  - TTFT = wall time of a 1-token greedy generation.
  - The answer is a separate 32-token greedy generation, scored with LongBench token F1. The prompt's
    `#` separators sometimes get echoed, so answers are cut at `#` (applied identically to every
    configuration).

## Answer quality (97 questions, median prompt 6.2k tokens)

| Config | F1 | Δ vs recompute |
|---|---|---|
| full recompute | 0.456 | – |
| LMCache `dev` (no reuse happens) | 0.459 | +0.3 |
| #5584 only (reuse, corrupted separator KV) | 0.405 | **−5.1** |
| **#5584 + #5595** | **0.440** | **−1.6** |

- This is within CacheBlend's stated "1–2 point" quality cost.
- #5595 recovers 3.5 of the 5.1 F1 points lost to the separator bug.
- The answer requests ran after the timed request, so they had full cache hits. F1 therefore reflects
  full reuse.

## TTFT with full cache hits (6 questions, paired on the same questions)

| Prompt tokens | recompute | LMCache `dev` | both fixes | vs recompute |
|---|---|---|---|---|
| 6,166 | 405 ms | 486 ms | 241 ms | 1.68× |
| 7,739–7,831 (3 q) | 508–521 ms | 598–601 ms | 226–228 ms | 2.23–2.30× |
| 11,298–11,589 (2 q) | 781–799 ms | 872–891 ms | 285–297 ms | 2.63–2.81× |

- The median is **2.28× vs recompute and 2.66× vs LMCache `dev`**. The speedup grows with prompt length.
- The sample is small (n = 6) because of the pin-leak stall described below.
- Over the 97-question run, TTFT was 408 → 315 ms (1.3×). Many timed requests had only partial hits
  there because of the save-skip bug below, so that figure understates the speedup.

## Two more LMCache bugs found by this workload (not yet filed)

1. **Short new segments aren't saved after any hit.** `num_saved_tokens` is initialized to the hit
   count, and saving is skipped until the prompt crosses the next 256-token chunk boundary. That rule
   suits prefix caching. Under blending, a short passage precomputed after a system-prompt hit is
   never stored, so later lookups stop there.
   - Benchmark workaround: start warm-up prompts with the separator, so there is no prior hit.
2. **Pin leak, then stall and deadlock.**
   - Lookups leave objects pinned ("Double unpin", then "Pin timeout ... Forcing unpin" after the
     300 s default).
   - Each request then stalls until the monitor fires. With a 5 s timeout the engine eventually
     deadlocked (0% GPU, no progress).
   - Excerpt: `results/pin_leak_excerpt.txt`.

## MuSiQue

- Only 6 questions fit under the 12k-token cap (20 passages per question), which is too few to report.
- The answers were coherent, and the F1 gap came from 1–2 questions.
- The run with a higher cap (15.5k) hit the pin-leak stall before finishing.

Raw data: `results/`. Scripts: `real_rag.py`, `run_realrag*.sh`, `score.py`.
