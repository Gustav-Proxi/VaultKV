"""Blend-vs-recompute output agreement on exact-repeat and reordered prompts.

Usage: python qrepro.py {recompute|lmcache} OUT.jsonl
Per trial (fresh docs): cold request (stores), then 32-token greedy generations
for an exact repeat and for the docs reordered. Debug/fix behavior comes from
LMC_* env vars (see patch_debug.py).
"""

import json
import os
import random
import re
import sys

MODE, OUT = sys.argv[1], sys.argv[2]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DOC_TOKENS = int(os.environ.get("DOC_TOKENS", "2000"))
N_DOCS = int(os.environ.get("N_DOCS", "3"))
TRIALS = int(os.environ.get("TRIALS", "3"))
SEP_STR = " # # "

BLEND = os.environ.get("NOBLEND") != "1"
if MODE == "lmcache":
    os.environ.update(
        {
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_ENABLE_BLENDING": "True" if BLEND else "False",
            "LMCACHE_BLEND_SPECIAL_STR": SEP_STR,
            "LMCACHE_USE_LAYERWISE": os.environ.get("LAYERWISE", "True"),
            "LMCACHE_BLEND_CHECK_LAYERS": "1",
            "LMCACHE_BLEND_RECOMPUTE_RATIOS": os.environ.get("RATIO", "0.15"),
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "20",
        }
    )

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

WORDS = (
    "revenue margin quarter forecast region supplier contract audit policy "
    "inventory shipment customer warranty compliance budget hiring roadmap"
).split()


def make_doc(rng: random.Random, tok, doc_id: int) -> list[int]:
    sentences, ids = [], []
    while len(ids) < DOC_TOKENS:
        a, b = rng.sample(WORDS, 2)
        sentences.append(
            f"Doc {doc_id} fact {len(sentences)}: the {a} figure for {b} was "
            f"{rng.randint(10, 9999)} units in week {rng.randint(1, 52)}."
        )
        ids = tok.encode(" ".join(sentences), add_special_tokens=False)
    return ids[:DOC_TOKENS]


FACT = re.compile(r"Doc (\d+) fact \d+: the \w+ figure for \w+ was (\d+) units")


def ground_truth(tok, docs: list[list[int]]) -> dict:
    """Largest units figure in the text the model actually sees (after truncation)."""
    best = {"doc": None, "value": -1}
    for ids in docs:
        for doc_id, value in FACT.findall(tok.decode(ids)):
            if int(value) > best["value"]:
                best = {"doc": int(doc_id), "value": int(value)}
    return best


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    kwargs = dict(
        model=MODEL,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    if MODE == "lmcache":
        from vllm.config import KVTransferConfig

        kwargs.update(
            enable_prefix_caching=False,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnectorV1", kv_role="kv_both"
            ),
        )
    else:
        kwargs.update(enable_prefix_caching=False)
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

    one = SamplingParams(temperature=0, max_tokens=1)
    gen = SamplingParams(temperature=0, max_tokens=32)

    def text(ids: list[int]) -> str:
        return llm.generate({"prompt_token_ids": ids}, gen, use_tqdm=False)[0].outputs[0].text

    with open(OUT, "w") as out:
        for trial in range(TRIALS + 1):  # trial 0 = warmup
            rng = random.Random(2000 + trial)
            docs = [make_doc(rng, tok, trial * 10 + d) for d in range(N_DOCS)]
            order = list(range(N_DOCS))[::-1]
            cold = join(system, *docs, question, suffix)
            reorder = join(system, *[docs[i] for i in order], question, suffix)
            llm.generate({"prompt_token_ids": cold}, one, use_tqdm=False)
            gt = ground_truth(tok, docs)
            row = {
                "gt": gt,
                "mode": MODE,
                "env": {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("LMC_") or k in ("NOBLEND", "LAYERWISE")
                },
                "trial": trial,
                "tokens": len(cold),
                "repeat_text": text(cold),
                "reorder_text": text(reorder),
                "warmup": trial == 0,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print("@@ROW", json.dumps({k: row[k] for k in ("mode", "trial", "tokens")}), flush=True)


if __name__ == "__main__":
    main()
