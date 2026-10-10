**What this PR does / why we need it**:

`SegmentTokenDatabase.process_tokens` yields an empty segment when two separator matches are adjacent. `store_layer` skips empty ranges (`if end <= start: continue`), so that key is never stored. `lookup` doesn't skip it: it treats the empty segment as a miss and stops matching. With blending enabled, lookup therefore never gets past the first segment, even on an exact repeat of a stored prompt.

This happens with the default separator on tokenizers without a BOS token. For example, `LMCACHE_BLEND_SPECIAL_STR=" # # "` is env-stripped to `"# #"`, and on Qwen2.5 the `[1:]` slice leaves the 1-token separator `" #"`. The `" # # "` in the prompt contains that token twice.

This PR stops `process_tokens` from yielding empty segments, so lookup and store see the same segments.

Measured on A100, vLLM 0.20.0, Qwen2.5-1.5B, ~3.1k-token prompt (details in #3238):

| Turn | before | after |
|---|---|---|
| exact repeat | 12 / 3161 hit tokens | 3161 / 3161 |
| document moved before the question | 12 / 3161 | 3161 / 3161 |
| new question, same document | 12 / 3161 | 12 / 3161 (prefix-only lookup, separate fix) |

Refs #3238. This fixes one of the two causes found there. The prefix-only lookup still needs a design discussion.

**Special notes for your reviewers**:

- This is behavior-preserving for prompts without adjacent separators: non-empty segments keep the same `(start, end, hash)`.
- The new test uses a fake no-BOS tokenizer, so it needs no HF credentials or GPU. It fails on `dev` and passes with this change.
- This complements #3228, which fixes `sep_tokens` tokenization. With #3228 merged, the default `" # # "` on Qwen no longer produces adjacent matches. Any prompt with two separators in a row still yields an empty segment, and this change keeps lookup and store consistent in that case. The two touch different lines. #3819 and #4584 also touch only `_fast_split_by_subtensor`.

**If applicable**:

- [ ] this PR contains user facing changes - docs added
- [x] this PR contains unit tests
