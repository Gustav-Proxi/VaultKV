# ruff: noqa  (one-off Colab patch script; anchors must match upstream text exactly)
"""vLLM 0.31: register the model for LMCache CacheBlend right after it loads.

KV-connector init already runs later (initialize_from_config), so only the
registration from LMCache's blend_in_process README is still needed.
"""

import vllm.v1.worker.gpu_worker as gw

path = gw.__file__
src = open(path).read()
MARK = "# LMCACHE_3238_PATCH"
anchor = "            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)\n"
if MARK in src:
    print("already patched", path)
else:
    assert src.count(anchor) == 1, "load_model anchor not found exactly once"
    add = (
        "        " + MARK + "\n"
        "        from lmcache.integration.vllm.utils import ENGINE_NAME\n"
        "        from lmcache.v1.compute.models.utils import VLLMModelTracker\n"
        "        VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)\n"
    )
    open(path, "w").write(src.replace(anchor, anchor + add))
    print("patched", path)
