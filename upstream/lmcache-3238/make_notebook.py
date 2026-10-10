# ruff: noqa: E501  (cell sources are notebook content)
"""Generate colab_repro_3238.ipynb: GPU reproduction of LMCache #3238 on vLLM V1.

Run: python upstream/lmcache-3238/make_notebook.py
"""

import json
from pathlib import Path

LMCACHE_COMMIT = "34d8027"  # dev commit traced on 2026-10-09
VLLM_VERSION = "0.18.0"  # version used in the original issue report

INTRO = f"""# LMCache #3238 — GPU reproduction (vLLM V1 + CacheBlend)

Confirms whether non-prefix chunk reuse works through the vLLM V1 connector and
records baseline TTFT numbers for the later fix.

**Runtime:** Runtime → Change runtime type → **A100** (preferred) or **L4**.

Pinned: vLLM `{VLLM_VERSION}`, LMCache `dev@{LMCACHE_COMMIT}` built from source.
The LMCache build compiles CUDA kernels and takes about 10–20 minutes.

Workload (from the issue): `[system] SEP [question] SEP [document] SEP`, where the
~2,300-token document is identical across turns and the question changes.

| Run | Prompt | Expected if the bug is present |
|---|---|---|
| turn 1 | sys + Q1 + doc | 0 hits (cold) |
| turn 2 | sys + **Q2** + doc | only ~system-prompt tokens hit |
| turn 3 | turn 2 repeated | near-full hit (positive control: LMCache works at all) |
| turn 4 | doc moved before Q2 | only ~system-prompt tokens hit |
"""

SETUP_GPU = """!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import sys; print(sys.version)"""

INSTALL_VLLM = f"""%%bash
set -e
pip install -q "vllm=={VLLM_VERSION}"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
"""

INSTALL_LMCACHE = f"""%%bash
set -e
cd /content
[ -d LMCache ] || git clone -q https://github.com/LMCache/LMCache.git
cd LMCache
git fetch -q origin dev
git checkout -q {LMCACHE_COMMIT}
git log -1 --format='LMCache %h %cd %s'
# A100 = 8.0, L4 = 8.9
export TORCH_CUDA_ARCH_LIST="8.0;8.9"
pip install -q -r requirements/build.txt
pip install -q -e . --no-build-isolation
python -c "import lmcache, lmcache.c_ops; print('lmcache ok')"
"""

PATCH_VLLM = """# CacheBlend in-process mode needs the model registered before KV transfer init
# (see LMCache examples/blend_in_process/README.md). Patch vLLM's gpu_worker.py.
import re, inspect, vllm.v1.worker.gpu_worker as gw

path = inspect.getfile(gw)
src = open(path).read()
MARK = "# LMCACHE_3238_PATCH"
if MARK in src:
    print("already patched:", path)
else:
    init_fn = re.search(r"def init_worker_distributed_environment\\(.*?(?=\\ndef |\\Z)", src, re.S)
    assert init_fn, "init_worker_distributed_environment not found; vLLM layout changed"
    body = init_fn.group(0)
    call = re.search(r"\\n([ \\t]*)ensure_kv_transfer_initialized\\(vllm_config\\)", body)
    assert call, "ensure_kv_transfer_initialized(vllm_config) not found in init fn"
    new_body = body.replace(call.group(0), f"\\n{call.group(1)}pass  {MARK} (moved to load_model)")
    src = src.replace(body, new_body)

    load = re.search(r"\\n([ \\t]*)def load_model\\(self.*?\\n(?=\\1def |\\Z)", src, re.S)
    assert load, "load_model not found"
    indent = load.group(1) + "    "
    addition = (
        f"\\n{indent}{MARK}\\n"
        f"{indent}from lmcache.v1.compute.models.utils import VLLMModelTracker\\n"
        f"{indent}from lmcache.integration.vllm.utils import ENGINE_NAME\\n"
        f"{indent}from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized\\n"
        f"{indent}VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)\\n"
        f"{indent}ensure_kv_transfer_initialized(self.vllm_config)\\n"
    )
    src = src.replace(load.group(0), load.group(0).rstrip("\\n") + addition + "\\n")
    open(path, "w").write(src)
    print("patched:", path)
"""

REPRO = '''%%writefile /content/repro_3238.py
"""Two-turn CacheBlend workload from LMCache #3238. Usage: python repro_3238.py {lmcache|apc}"""
import os, sys, time, json

MODE = sys.argv[1]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SEP_STR = " # # "

if MODE == "lmcache":
    os.environ.update({
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_ENABLE_BLENDING": "True",
        "LMCACHE_BLEND_SPECIAL_STR": SEP_STR,
        "LMCACHE_USE_LAYERWISE": "True",
        "LMCACHE_BLEND_CHECK_LAYERS": "1",
        "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
    })

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained(MODEL)
enc = lambda s: tok.encode(s, add_special_tokens=False)

kwargs = dict(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.6, enforce_eager=True)
if MODE == "lmcache":
    from vllm.config import KVTransferConfig
    kwargs.update(enable_prefix_caching=False,
                  kv_transfer_config=KVTransferConfig(kv_connector="LMCacheConnectorV1",
                                                      kv_role="kv_both"))
else:
    kwargs.update(enable_prefix_caching=True)
llm = LLM(**kwargs)

sep = enc(SEP_STR)
system = enc("You are a careful assistant. Answer using only the document.")
doc = enc(" ".join(f"Section {i}: the quarterly report lists item {i} with value {i * 37 % 101}."
                   for i in range(160)))
q1 = enc("What is the value of item 12?")
q2 = enc("Which item has the largest value, and what is it?")
j = lambda *parts: sum(([*p, *sep] for p in parts[:-1]), []) + list(parts[-1])

prompts = {
    "warmup": enc("Nice to meet you. " * 300),
    "turn1_cold": j(system, q1, doc),
    "turn2_new_question": j(system, q2, doc),
    "turn3_exact_repeat": j(system, q2, doc),
    "turn4_doc_reordered": j(system, doc, q2),
}
sp = SamplingParams(temperature=0, max_tokens=1)
results = []
for name, ids in prompts.items():
    t0 = time.perf_counter()
    llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    print(f"@@RESULT {json.dumps({'mode': MODE, 'turn': name, 'tokens': len(ids), 'ttft_s': round(dt, 4)})}", flush=True)
    time.sleep(1)
print(f"@@DOC_TOKENS {len(doc)} SYSTEM_TOKENS {len(system)}")
'''

RUN = """%%bash
cd /content
for mode in apc lmcache; do
  echo "===== $mode ====="
  python repro_3238.py $mode 2>&1 | tee run_$mode.log | grep -E "@@|LMCache hit tokens|Traceback|Error" | tail -40
done
"""

SUMMARY = """import json, re
rows = []
for mode in ("apc", "lmcache"):
    log = open(f"/content/run_{mode}.log").read()
    hits = [int(h) for h in re.findall(r"LMCache hit tokens: (\\d+)", log)]
    res = [json.loads(m) for m in re.findall(r"@@RESULT (.*)", log)]
    for i, r in enumerate(res):
        r["lmcache_hit_tokens"] = hits[i] if mode == "lmcache" and i < len(hits) else None
        rows.append(r)
print(f"{'mode':8} {'turn':22} {'tokens':>6} {'ttft_s':>8} {'lmcache_hits':>12}")
for r in rows:
    print(f"{r['mode']:8} {r['turn']:22} {r['tokens']:>6} {r['ttft_s']:>8} {str(r['lmcache_hit_tokens']):>12}")
print(re.search(r"@@DOC_TOKENS.*", open('/content/run_lmcache.log').read()).group(0))
"""

INTERPRET = """## Reading the result

- **Bug reproduced** if `turn2_new_question` and `turn4_doc_reordered` show `lmcache_hits`
  ≈ system-prompt tokens while `turn3_exact_repeat` shows a near-full hit.
- `ttft_s` here is wall time for a 1-token generation (≈ prefill time). These are the
  **"before" numbers**; the fix should drop turn 2/4 toward turn 3's time.
- Copy the summary table and the `nvidia-smi` line into the experiment notes. If a cell fails, copy
  the traceback (the logs are in `/content/run_*.log`).
"""


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "cells": [
        cell("markdown", INTRO),
        cell("code", SETUP_GPU),
        cell("code", INSTALL_VLLM),
        cell("code", INSTALL_LMCACHE),
        cell("code", PATCH_VLLM),
        cell("code", REPRO),
        cell("code", RUN),
        cell("code", SUMMARY),
        cell("markdown", INTERPRET),
    ],
}

out = Path(__file__).with_name("colab_repro_3238.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
