# ruff: noqa  (one-off Colab patch script; anchors must match upstream text exactly)
"""Log every layer the layerwise loader flushes into vLLM's paged KV cache."""

p = "/content/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py"
s = open(p).read()
old = "                del self.buffer_mapping[layer_id - 2]\n"
new = old + "                logger.info('DBG3238 flushed layer %d', layer_id - 2)\n"
if "DBG3238 flushed" not in s:
    assert s.count(old) == 1, "flush anchor"
    open(p, "w").write(s.replace(old, new))
print("flush logging installed")
