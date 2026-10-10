# ruff: noqa  (one-off Colab patch script; anchors must match upstream text exactly)
"""Candidate fix 1: SegmentTokenDatabase.process_tokens must not yield empty segments."""

p = "/content/LMCache/lmcache/v1/token_database.py"
s = open(p).read()
old = "                if start_idx >= num_falses:\n                    if make_key:\n                        yield (\n                            start_idx,\n                            end_idx,\n                            self._make_key_by_hash(\n                                self._hash_tokens(token_chunk), request_configs\n"
new = "                # Adjacent separators produce empty segments; store skips them,\n                # so lookup must too or it reports a miss and stops early.\n                if token_chunk_len > 0 and start_idx >= num_falses:\n                    if make_key:\n                        yield (\n                            start_idx,\n                            end_idx,\n                            self._make_key_by_hash(\n                                self._hash_tokens(token_chunk), request_configs\n"
if "Adjacent separators" not in s:
    assert s.count(old) == 1, "anchor not found"
    open(p, "w").write(s.replace(old, new))
print("fix1 applied")
