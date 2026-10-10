I re-traced this on current `dev` (34d8027) and can still reproduce the 24-token symptom, but the cause seems to have moved.

- Segment hashing is now position-independent: `SegmentTokenDatabase.process_tokens` hashes each segment by its own content, as @andyluo7 noted.
- `LMCacheEngine.lookup` → `StorageManager.batched_contains` still returns a **contiguous-prefix** hit count. Lookup stops at the first missed segment (the changed question) and returns the system-prompt length, even though the document segment's key is present.
- On the worker side, `blend()` is also only called on `tokens[:lmcache_cached_tokens]`.

A CPU-only repro (no GPU or tokenizer download) runs the real `lookup`, `SegmentTokenDatabase.process_tokens` and `batched_contains` against an in-memory backend, with only the compiled extensions stubbed:

```
turn2 tokens:            2358
segments (start,end):    [(0, 24), (27, 55), (58, 2358)]
doc segment cached:      True  (2300 tokens)
lookup() hit tokens:     24
actually reusable:       2324
```

<details><summary>repro_3238_lookup.py</summary>

```python
"""CPU reproduction of LMCache #3238 at the lookup layer (no GPU, no tokenizer download).

Runs the real LMCacheEngine.lookup, SegmentTokenDatabase.process_tokens and
StorageManager.batched_contains from an LMCache source tree. Only the compiled
CUDA extension and engine construction are stubbed.

Usage: python -I repro_3238_lookup.py /path/to/LMCache
"""

import sys
from unittest.mock import MagicMock

sys.path.insert(0, sys.argv[1])

# Compiled extensions are irrelevant to lookup; stub them before importing lmcache.
import lmcache  # noqa: E402

for name in ("device_ops", "c_ops", "lmcache_native", "native_storage_ops"):
    stub = MagicMock(name=f"lmcache.{name}")
    sys.modules[f"lmcache.{name}"] = stub
    setattr(lmcache, name, stub)

import torch  # noqa: E402
from lmcache.v1.cache_engine import LMCacheEngine  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.storage_backend.storage_manager import StorageManager  # noqa: E402
from lmcache.v1.token_database import SegmentTokenDatabase  # noqa: E402

SEP = torch.tensor([90001, 90002, 90003])  # stand-in for tokenized blend_special_str


def make_segment_db() -> SegmentTokenDatabase:
    """Build a real SegmentTokenDatabase without downloading a tokenizer."""
    from lmcache.v1.token_database import TokenDatabase

    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ", chunk_size=256)
    meta = MagicMock()
    meta.model_name = "test_model"
    meta.world_size = 1
    meta.worker_id = 0
    meta.kv_dtype = torch.bfloat16
    meta.role = "worker"
    db = SegmentTokenDatabase.__new__(SegmentTokenDatabase)
    TokenDatabase.__init__(db, cfg, meta)
    db.sep_tokens = SEP
    db.sep_len = len(SEP)
    return db


class DictBackend:
    """Mirrors LocalCPUBackend.batched_contains: count of leading keys present."""

    def __init__(self) -> None:
        self.keys = set()

    def batched_contains(self, keys, pin=False):
        n = 0
        for key in keys:
            if key not in self.keys:
                break
            n += 1
        return n


def make_engine(db, backend):
    sm = StorageManager.__new__(StorageManager)
    sm.get_active_storage_backends = lambda search_range=None: [("LocalCPUBackend", backend)]
    sm.touch_cache = lambda: None
    eng = LMCacheEngine.__new__(LMCacheEngine)
    eng.token_database = db
    eng.storage_manager = sm
    eng.use_layerwise = False
    eng.retrieve_locations = None
    eng.lookup_pins = {}
    eng.is_healthy = lambda: True
    eng.stats_monitor = MagicMock()
    return eng


def segs(*parts):
    out = [parts[0]]
    for p in parts[1:]:
        out += [SEP, p]
    return torch.cat(out)


def rand(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 50000, (n,), generator=g)


def main() -> int:
    db = make_segment_db()
    backend = DictBackend()
    eng = make_engine(db, backend)

    system, q1, q2, doc = rand(24, 1), rand(30, 2), rand(28, 3), rand(2300, 4)
    turn1 = segs(system, q1, doc)
    turn2 = segs(system, q2, doc)

    # "Store" turn 1: every segment key it produces is now cached.
    for _, _, key in db.process_tokens(tokens=turn1):
        backend.keys.add(key)

    t2 = list(db.process_tokens(tokens=turn2))
    doc_start, doc_end, doc_key = t2[-1]
    hits = eng.lookup(tokens=turn2)

    print(f"turn2 tokens:            {len(turn2)}")
    print(f"segments (start,end):    {[(s, e) for s, e, _ in t2]}")
    print(f"doc segment cached:      {doc_key in backend.keys}  ({doc_end - doc_start} tokens)")
    print(f"lookup() hit tokens:     {hits}")
    reusable = sum(e - s for s, e, k in t2 if k in backend.keys)
    print(f"actually reusable:       {reusable}")
    if doc_key in backend.keys and hits <= len(system):
        print(
            "REPRODUCED: document KV is cached and its key matches, but lookup stops "
            "at the first missed segment and reports only the system-prompt prefix."
        )
        return 1
    print("NOT reproduced: lookup reports non-prefix segment hits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

</details>

A fix seems to need three coordinated changes: non-prefix hit reporting in lookup for blending mode, masked retrieval that skips missing chunks, and forced recompute of missed segments in the blender. Because vLLM V1 reads `get_num_new_matched_tokens` as "the first N tokens are computed", I'd like to check the intended approach with maintainers before writing it.

@yaojiejia are you still working on this? I'm happy to collaborate, or to take it if not. I'm setting up a GPU reproduction on vLLM V1 and can share benchmarks.
