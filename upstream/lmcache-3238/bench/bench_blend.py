"""TTFT benchmark for CacheBlend non-prefix reuse (LMCache #3238 / PR #5584).

Usage: python bench_blend.py MODE OUT.jsonl
  MODE: recompute | apc | lmcache   (lmcache = LMCacheConnectorV1 with blending)

Each trial uses fresh documents:
  cold      sys | D1 | D2 | D3 | D4 | Q   (stores segments)
  reorder   sys | D3 | D1 | D4 | D2 | Q   (same documents, new order)
  repeat    sys | D1 | D2 | D3 | D4 | Q   (exact repeat of cold)
TTFT = wall time of a 1-token greedy generation. Then a 32-token greedy
generation of the reorder prompt is recorded for quality comparison.
"""

import json
import os
import random
import sys
import time

MODE, OUT = sys.argv[1], sys.argv[2]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
DOC_TOKENS = int(os.environ.get("DOC_TOKENS", "3000"))
N_DOCS = int(os.environ.get("N_DOCS", "4"))
TRIALS = int(os.environ.get("TRIALS", "5"))
SEP_STR = " # # "

if MODE == "lmcache":
    os.environ.update(
        {
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_ENABLE_BLENDING": "True",
            "LMCACHE_BLEND_SPECIAL_STR": SEP_STR,
            "LMCACHE_USE_LAYERWISE": "True",
            "LMCACHE_BLEND_CHECK_LAYERS": "1",
            "LMCACHE_BLEND_RECOMPUTE_RATIOS": os.environ.get("RATIO", "0.15"),
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "30",
        }
    )

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

WORDS = (
    "revenue margin quarter forecast region supplier contract audit policy "
    "inventory shipment customer warranty compliance budget hiring roadmap "
    "incident latency capacity migration vendor invoice renewal pricing"
).split()


def make_doc(rng: random.Random, tok, doc_id: int) -> list[int]:
    """Synthetic factual document of ~DOC_TOKENS tokens (unique per call)."""
    sentences, ids = [], []
    while len(ids) < DOC_TOKENS:
        a, b = rng.sample(WORDS, 2)
        sentences.append(
            f"Doc {doc_id} fact {len(sentences)}: the {a} figure for {b} was "
            f"{rng.randint(10, 9999)} units in week {rng.randint(1, 52)}."
        )
        ids = tok.encode(" ".join(sentences), add_special_tokens=False)
    return ids[:DOC_TOKENS]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    kwargs = dict(model=MODEL, max_model_len=32768, gpu_memory_utilization=0.85, enforce_eager=True)
    if os.environ.get("MAX_BATCHED"):
        # Chunked prefill splits a segment across steps and stores it in pieces
        # whose keys never match the whole segment; one step per prompt avoids it.
        kwargs["max_num_batched_tokens"] = int(os.environ["MAX_BATCHED"])
    if MODE == "lmcache":
        from vllm.config import KVTransferConfig

        kwargs.update(
            enable_prefix_caching=False,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnectorV1", kv_role="kv_both"
            ),
        )
    else:
        kwargs.update(enable_prefix_caching=(MODE == "apc"))
    llm = LLM(**kwargs)

    sep = enc(SEP_STR)
    system = enc("You are a precise analyst. Answer only from the documents below.")
    question = enc("Question: Which document mentions the largest units figure, and what is it?")
    suffix = enc("Answer:")

    def join(*parts: list[int]) -> list[int]:
        out: list[int] = []
        for part in parts[:-1]:
            out += part + sep
        return out + parts[-1]

    ttft_params = SamplingParams(temperature=0, max_tokens=1)
    gen_params = SamplingParams(temperature=0, max_tokens=32)

    def timed(ids: list[int]) -> float:
        start = time.perf_counter()
        llm.generate({"prompt_token_ids": ids}, ttft_params, use_tqdm=False)
        return time.perf_counter() - start

    with open(OUT, "w") as out:
        # Trial 0 warms up kernels and the blend code path; it is not reported.
        for trial in range(TRIALS + 1):
            rng = random.Random(1000 + trial)
            docs = [make_doc(rng, tok, trial * 10 + d) for d in range(N_DOCS)]
            order = list(range(N_DOCS))
            rng.shuffle(order)
            if order == list(range(N_DOCS)):
                order = order[1:] + order[:1]
            cold = join(system, *docs, question, suffix)
            reorder = join(system, *[docs[i] for i in order], question, suffix)

            row = {"mode": MODE, "trial": trial, "prompt_tokens": len(cold)}
            row["cold_s"] = timed(cold)
            time.sleep(1)
            row["reorder_s"] = timed(reorder)
            time.sleep(1)
            row["repeat_s"] = timed(cold)
            time.sleep(1)
            gen = llm.generate({"prompt_token_ids": reorder}, gen_params, use_tqdm=False)
            row["reorder_text"] = gen[0].outputs[0].text
            # Exact repeat keeps every token at its stored position, so blended
            # output should match full recompute; divergence here means a bug.
            gen = llm.generate({"prompt_token_ids": cold}, gen_params, use_tqdm=False)
            row["repeat_text"] = gen[0].outputs[0].text
            row["warmup"] = trial == 0
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"@@ROW {json.dumps({k: v for k, v in row.items() if not k.endswith('_text')})}")


if __name__ == "__main__":
    main()
