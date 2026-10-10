"""CacheBlend on real RAG data: LongBench 2WikiMQA / MuSiQue, TTFT + token F1.

Usage: python real_rag.py {recompute|lmcache} DATASET OUT.jsonl
  DATASET: path to a LongBench jsonl (e.g. /content/data/2wikimqa.jsonl)

Per sample, following CacheBlend's evaluation: every passage is first sent
alone (precomputed chunk KV), then the full prompt
  instruction | P1 | ... | Pn | question | "Answer:"
is answered. TTFT = wall time of a 1-token greedy generation; the answer is a
separate 32-token greedy generation scored with LongBench's qa_f1_score.
"""

import json
import os
import re
import string
import sys
import time
from collections import Counter

MODE, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_PROMPT = int(os.environ.get("MAX_PROMPT", "12000"))
LIMIT = int(os.environ.get("LIMIT", "200"))
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
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "30",
        }
    )

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

INSTRUCTION = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\nThe following are given passages."
)


def normalize(s: str) -> str:
    """LongBench / SQuAD answer normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def split_passages(context: str) -> list[str]:
    parts = re.split(r"Passage \d+:\n", context)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    kwargs = dict(
        model=MODEL,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85,
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
    system = enc(INSTRUCTION)
    suffix = enc("Answer:")

    def join(*parts: list[int]) -> list[int]:
        out: list[int] = []
        for part in parts[:-1]:
            out += part + sep
        return out + parts[-1]

    one = SamplingParams(temperature=0, max_tokens=1)
    gen = SamplingParams(temperature=0, max_tokens=32)

    rows = [json.loads(line) for line in open(DATA)]
    done = 0
    with open(OUT, "w") as out:
        for i, row in enumerate(rows):
            if done >= LIMIT:
                break
            passages = [enc(p) for p in split_passages(row["context"])]
            question = enc(
                "Answer the question based on the given passages. Only give me the "
                f"answer and do not output any other words.\n\nQuestion: {row['input']}"
            )
            prompt = join(system, *passages, question, suffix)
            if len(prompt) > MAX_PROMPT or not passages:
                continue
            # Precompute each passage's KV on its own (position-independent reuse).
            for p in passages:
                if os.environ.get("WARM_STYLE") == "sep":
                    # Leading SEP: same segment boundaries as the real prompt, but no
                    # system-prompt hit, so LMCache's chunk-boundary save skip (which
                    # drops short new segments after any hit) does not apply.
                    warm = sep + join(p, suffix)
                else:
                    warm = join(system, p, suffix)
                llm.generate({"prompt_token_ids": warm}, one, use_tqdm=False)
            start = time.perf_counter()
            llm.generate({"prompt_token_ids": prompt}, one, use_tqdm=False)
            ttft = time.perf_counter() - start
            answer = (
                llm.generate({"prompt_token_ids": prompt}, gen, use_tqdm=False)[0]
                .outputs[0]
                .text.strip()
                .split("\n")[0]
            )
            f1 = max(qa_f1(answer, g) for g in row["answers"])
            rec = {
                "mode": MODE,
                "idx": i,
                "tokens": len(prompt),
                "n_passages": len(passages),
                "ttft_s": ttft,
                "answer": answer,
                "gold": row["answers"],
                "f1": f1,
                "warmup": done < 3,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done += 1
            if done % 20 == 0:
                print(f"@@PROGRESS {done}", flush=True)


if __name__ == "__main__":
    main()
