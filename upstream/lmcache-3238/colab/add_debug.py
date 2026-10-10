# ruff: noqa  (one-off Colab patch script; anchors must match upstream text exactly)
"""Temporary #3238 debug logging in the Colab LMCache checkout (cache_engine.py)."""

p = "/content/LMCache/lmcache/v1/cache_engine.py"
s = open(p).read()
if "DBG3238" not in s:
    # lookup, layerwise branch: log every segment and whether all layers hit
    a = "                    if hit_chunks == self.num_layers and len(block_mapping) == 1:\n"
    assert s.count(a) == 1
    s = s.replace(
        a,
        "                    logger.info('DBG3238 lookup seg [%d,%d) hash=%s hit_layers=%d/%d locs=%d', start, end, str(key.chunk_hash)[:12], hit_chunks, self.num_layers, len(block_mapping))\n"
        + a,
    )
    # store_layer: log each segment and whether it is skipped as already present
    b = "            keys_multi_layer = key.split_layers(self.num_layers)\n            # Only check the first layer\n"
    assert s.count(b) == 1
    s = s.replace(
        b,
        b
        + "            logger.info('DBG3238 store seg [%d,%d) hash=%s exists=%s', start, end, str(key.chunk_hash)[:12], self.storage_manager.contains(keys_multi_layer[0], self.retrieve_locations))\n",
    )
    open(p, "w").write(s)
print("debug added")
