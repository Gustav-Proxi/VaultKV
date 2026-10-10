"""GPU reproduction of LMCache #3238 on vLLM V1 with CacheBlend.

Usage: python repro_3238_gpu.py {lmcache|apc}
  lmcache: LMCacheConnectorV1 with blending (vLLM prefix caching off)
  apc:     vLLM automatic prefix caching only (baseline)

Prints one '@@RESULT {json}' line per turn; LMCache logs 'LMCache hit tokens: N'.
"""

import json
import os
import sys
import time

MODE = sys.argv[1]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SEP_STR = " # # "

if MODE == "lmcache":
    os.environ.update(
        {
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_ENABLE_BLENDING": "True",
            "LMCACHE_BLEND_SPECIAL_STR": SEP_STR,
            "LMCACHE_USE_LAYERWISE": "True",
            "LMCACHE_BLEND_CHECK_LAYERS": "1",
            "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
        }
    )

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    kwargs = dict(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.6, enforce_eager=True)
    if os.environ.get("ATTN"):
        # LMCache's layerwise (blend) kernel needs a 5-D per-layer KV tensor.
        kwargs["attention_backend"] = os.environ["ATTN"]
    if MODE == "lmcache":
        from vllm.config import KVTransferConfig

        kwargs.update(
            enable_prefix_caching=False,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnectorV1", kv_role="kv_both"
            ),
        )
    else:
        kwargs.update(enable_prefix_caching=True)
    llm = LLM(**kwargs)

    sep = enc(SEP_STR)
    system = enc("You are a careful assistant. Answer using only the document.")
    doc = enc(
        " ".join(
            f"Section {i}: the quarterly report lists item {i} with value {i * 37 % 101}."
            for i in range(160)
        )
    )
    q1 = enc("What is the value of item 12?")
    q2 = enc("Which item has the largest value, and what is it?")
    # The last segment's KV is stored truncated, so (as in the issue) every reusable
    # segment is followed by SEP and the prompt ends with a short suffix.
    suffix = enc("Answer:")

    def join(*parts: list[int]) -> list[int]:
        out: list[int] = []
        for part in parts[:-1]:
            out += part + sep
        return out + parts[-1]

    prompts = {
        "warmup": enc("Nice to meet you. " * 300),
        "turn1_cold": join(system, q1, doc, suffix),
        "turn2_new_question": join(system, q2, doc, suffix),
        "turn3_exact_repeat": join(system, q2, doc, suffix),
        "turn4_doc_reordered": join(system, doc, q2, suffix),
    }
    params = SamplingParams(temperature=0, max_tokens=1)
    for name, ids in prompts.items():
        start = time.perf_counter()
        llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        row = {"mode": MODE, "turn": name, "tokens": len(ids), "ttft_s": round(elapsed, 4)}
        print(f"@@RESULT {json.dumps(row)}", flush=True)
        time.sleep(1)
    print(f"@@SIZES doc={len(doc)} system={len(system)} sep={len(sep)}", flush=True)


if __name__ == "__main__":
    main()
