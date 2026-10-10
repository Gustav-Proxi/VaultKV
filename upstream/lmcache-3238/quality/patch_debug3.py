# ruff: noqa  (one-off Colab patch script; anchors must match upstream text exactly)
"""LMC_GAPFIX=1: recompute separator (gap) positions instead of leaving their KV at 0.

- Before the check layer, the blender computes K/V for every token; copy the
  gap rows into the loaded buffer.
- At the check layer, force gap positions into the recompute set.
- With LMC_DBG_NOWRITE=1, write back only the gap rows.
"""

p = "/content/LMCache/lmcache/v1/compute/blend/blender.py"
s = open(p).read()
if "GAPFIX" not in s:
    a = "        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)\n"
    assert s.count(a) == 1, "rotary anchor"
    s = s.replace(
        a,
        a
        + """        import os as _os2  # GAPFIX
        _gapfix = _os2.environ.get("LMC_GAPFIX") == "1"
        _gaps = getattr(self.gpu_connector, "current_gap_positions", None)
        if _gapfix and _gaps is not None and _gaps.numel() and self.metadata.imp_indices is None:
            old_k[_gaps] = k[_gaps]
            old_v[_gaps] = v[_gaps]
""",
    )
    b = "            top_indices, _ = torch.sort(top_indices)\n"
    assert s.count(b) == 1, "sort anchor"
    s = s.replace(
        b,
        b
        + """            if _gapfix and _gaps is not None and _gaps.numel():
                top_indices = torch.unique(torch.cat([top_indices, _gaps.to(top_indices.device)]))
                topk_num = top_indices.numel()
""",
    )
    c = """        if self.metadata.imp_indices is not None and _os.environ.get("LMC_DBG_NOWRITE") == "1":
            return q, old_k, old_v, residual, attn_output, attn_metadata
"""
    assert s.count(c) == 1, "nowrite anchor"
    s = s.replace(
        c,
        """        if self.metadata.imp_indices is not None and _os.environ.get("LMC_DBG_NOWRITE") == "1":
            if _gapfix and _gaps is not None and _gaps.numel():
                _m = torch.isin(self.metadata.imp_indices, _gaps.to(self.metadata.imp_indices.device))
                old_k[self.metadata.imp_indices[_m]] = k[_m]
                old_v[self.metadata.imp_indices[_m]] = v[_m]
            return q, old_k, old_v, residual, attn_output, attn_metadata
""",
    )
    open(p, "w").write(s)
print("gapfix installed")
