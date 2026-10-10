# ruff: noqa  (one-off Colab patch script; anchors must match upstream text exactly)
"""Env-gated debug/fix hooks for the CacheBlend quality investigation.

LMC_DBG_LOG=1     log relative error of recomputed K vs loaded K per layer
LMC_DBG_NOWRITE=1 never write recomputed K/V back into the loaded buffer
LMC_FIX_CAUSAL=1  attention for selected (non-suffix) queries with a true
                  causal mask: query at position p sees keys [0, p]
"""

ROOT = "/content/LMCache/lmcache/v1/compute"

# 1) metadata: remember which absolute positions the selected queries have
p = f"{ROOT}/attention/metadata.py"
s = open(p).read()
old = "    def update_from_top_indices(self, top_indices: torch.Tensor):\n        top_k_num = len(top_indices)\n"
new = "    def update_from_top_indices(self, top_indices: torch.Tensor):\n        self.top_indices = top_indices  # DBG3238\n        top_k_num = len(top_indices)\n"
if "DBG3238" not in s:
    assert s.count(old) == 1, "metadata anchor"
    s = s.replace(old, new)
    open(p, "w").write(s)

# 2) flash-attn backend: optional true-causal path for selected queries
p = f"{ROOT}/attention/flash_attn.py"
s = open(p).read()
old = "        assert isinstance(attn_metadata, LMCFlashAttnMetadata)\n"
new = (
    old
    + """        import os as _os  # DBG3238
        _top = getattr(attn_metadata, "top_indices", None)
        if _os.environ.get("LMC_FIX_CAUSAL") == "1" and _top is not None and query.shape[0] < key.shape[0]:
            import torch.nn.functional as _F
            # flash-attn aligns causal masks bottom-right when seqlen_q != seqlen_k,
            # which lets scattered selected queries see future keys. Use an
            # explicit mask on true positions instead.
            n_k = key.shape[0]
            mask = torch.arange(n_k, device=key.device)[None, :] <= _top[:, None]
            q_ = query.transpose(0, 1).unsqueeze(0)
            k_ = key.transpose(0, 1).unsqueeze(0)
            v_ = value.transpose(0, 1).unsqueeze(0)
            out = _F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=mask, scale=self.vllm_attn_impl.scale, enable_gqa=True
            )
            output.copy_(out.squeeze(0).transpose(0, 1))
            return output
"""
)
if "DBG3238" not in s:
    assert s.count(old) == 1, "flash_attn anchor"
    s = s.replace(old, new)
    if "import torch\n" not in s:
        s = "import torch\n" + s
    open(p, "w").write(s)

# 3) blender: optional error logging and no-write ablation
p = f"{ROOT}/blend/blender.py"
s = open(p).read()
old = "        if self.metadata.imp_indices is not None:\n            old_k[self.metadata.imp_indices] = k\n            old_v[self.metadata.imp_indices] = v\n"
new = (
    """        import os as _os  # DBG3238
        if self.metadata.imp_indices is not None and _os.environ.get("LMC_DBG_LOG") == "1":
            _ok = old_k[self.metadata.imp_indices].float()
            _rel = ((k.float() - _ok).norm() / (_ok.norm() + 1e-6)).item()
            logger.info("DBG3238 layer %d recompute_rel_err_k=%.4f n=%d", layer_id, _rel, k.shape[0])
        if self.metadata.imp_indices is not None and _os.environ.get("LMC_DBG_NOWRITE") == "1":
            return q, old_k, old_v, residual, attn_output, attn_metadata
"""
    + old
)
if "DBG3238" not in s:
    assert s.count(old) == 1, "blender anchor"
    s = s.replace(old, new)
    open(p, "w").write(s)
print("debug hooks installed")
