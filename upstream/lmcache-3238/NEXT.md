# Done 2026-10-10: CacheBlend output-quality bug

**Result:**
- **Root cause:** separator (gap) KV is zero-filled by the blend loader and never recomputed. Fixed in
  [PR #5595](https://github.com/LMCache/LMCache/pull/5595).
- **Second bug:** the flash-attn bottom-right causal mask on scattered recomputed queries. Filed as
  [issue #5596](https://github.com/LMCache/LMCache/issues/5596) with a fix proposal.
- **Write-up:** `quality/FINDINGS_QUALITY.md`.

**Next:**
- Watch #5584, #5595 and #5596 for maintainer replies.
- If #5596's approach is accepted, implement the paged-KV `seqused_k` causal fix and benchmark it
  (TTFT plus agreement with recompute).
- Still unfiled: the chunked-prefill segment split.

---

## Original plan (kept for history)

# Next session: root-cause the CacheBlend output-quality bug

**Status at handoff (2026-10-09):**
- PR [#5584](https://github.com/LMCache/LMCache/pull/5584) (empty-segment fix) is open with CI green.
  The benchmark comment is posted. #5583 was closed (it had attribution; see memory).
- Blended outputs diverge from full recompute **even on exact repeats** ("the the" artifacts), at
  ratio 0.15 and at 0.5. See `bench/BENCHMARK.md` and `bench/results_quality/`.
- `dev` never exercised the blend path (no hits), so this is a pre-existing bug that #5584 exposes.

## Hypotheses, cheapest test first

1. **The last token is processed twice.** On a full hit the connector reports
   `need_to_allocate = hits - 1` (vLLM recomputes the last token), while `blend()` runs on
   `tokens[:lmcache_cached_tokens]` including that token.
   - Test: an exact repeat with `skip_last_n_tokens=1` (config exists in `vllm_v1_adapter`), or log
     slot_mapping / positions for the last token on both paths.
2. **Separator KV is never stored or loaded.** `Retrieved 12036 out of 12048` = 6 separators × 2
   tokens. Those slots keep stale KV unless they fall in the recompute top-k.
   - Test: force their indices into `imp_indices`.
   - Test: put a separator-free prompt (a single long segment + suffix) through the blend path. If it
     matches recompute, separators are the cause.
3. **Positions / RoPE.** `process_qkv` applies `rotary_emb` with `positions = arange(len)` and compares
   it with `old_k`, which was stored with RoPE at its original position.
   - Test: for an exact repeat, check that `diff_k` at layer 1 is ~0 for loaded tokens. A large diff
     everywhere means a RoPE or layout mismatch.
4. **vLLM 0.20 vs blend-path assumptions** (written against older vLLM). Check the KV layout from
   `gpu_connector.get_kv(layer_id)` against the layout vLLM 0.20 writes (NHD vs HND, the K/V split).

## Method

- **Minimal harness.** Qwen2.5-1.5B (fast) plus 7B to confirm. Compare greedy 32-token outputs and
  first-token logits (blend vs recompute) on an exact repeat. A logit L2 / top-1 agreement metric beats
  eyeballing text.
- **Debug hooks.** Add temporary logging in `blender.py` / `vllm_v1_adapter.py` on Colab, in the same
  style as `colab/add_debug.py`.
- **Colab workflow.** Use `colab new -s q --gpu A100`, upload `bench/setup_and_run.sh` (adapted),
  `colab/patch_vllm.py` and the scripts. Run with `nohup` and poll with `colab download` of logs.
  `colab exec` times out after ~2 min. Stop the session when done.
- **Upstream etiquette.** Review every comment, issue or PR before posting. Use DCO sign-off
  (`git commit -s`), branch from `dev`, and push through a contributor fork.

## Also pending

- File separate issues for the blend quality bug and for the chunked-prefill segment split (offered
  on the PR). Wait for a maintainer reply first, or file if there's none in a few days.
- Fix 2 (prefix-only lookup) needs a maintainer design decision. Watch #3238.
- Watch #5584 for review comments.
