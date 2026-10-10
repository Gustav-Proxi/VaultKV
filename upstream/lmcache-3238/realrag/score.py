"""Score real-RAG runs: TTFT, F1 (separator echo stripped), paired over shared samples.

Usage: python score.py RESULTS_DIR
"""

import json
import re
import statistics as st
import string
import sys
from collections import Counter
from pathlib import Path


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def clean(answer: str) -> str:
    # The prompt's " # # " separators sometimes get echoed; cut there.
    return answer.split("#")[0].strip()


def load(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.open():
        r = json.loads(line)
        if not r["warmup"]:
            r["f1c"] = max(qa_f1(clean(r["answer"]), g) for g in r["gold"])
            rows[r["idx"]] = r
    return rows


def main() -> None:
    d = Path(sys.argv[1])
    for ds in ("2wikimqa", "musique"):
        runs = {p.stem.split(f"rr_{ds}_")[1]: load(p) for p in sorted(d.glob(f"rr_{ds}_*.jsonl"))}
        if not runs:
            continue
        shared = set.intersection(*(set(r) for r in runs.values()))
        print(f"\n== {ds}: {len(shared)} shared samples")
        base = runs.get("recompute")
        tokens = st.median(base[i]["tokens"] for i in shared) if base else 0
        print(f"median prompt tokens: {tokens:.0f}")
        header = ("config", "TTFT p50", "TTFT mean", "F1", "F1 raw", "=recompute ans")
        print("{:10} {:>9} {:>10} {:>6} {:>7} {:>15}".format(*header))
        for name in ("recompute", "dev", "e5584", "both"):
            if name not in runs:
                continue
            r = runs[name]
            ttft = [r[i]["ttft_s"] * 1000 for i in shared]
            f1 = st.mean(r[i]["f1c"] for i in shared)
            f1raw = st.mean(r[i]["f1"] for i in shared)
            same = (
                sum(clean(r[i]["answer"]) == clean(base[i]["answer"]) for i in shared)
                if base
                else 0
            )
            print(
                f"{name:10} {st.median(ttft):8.0f}ms {st.mean(ttft):9.0f}ms "
                f"{f1:6.3f} {f1raw:7.3f} {same:>9}/{len(shared)}"
            )


if __name__ == "__main__":
    main()
