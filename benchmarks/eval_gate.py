"""Eval gate: block merges on retrieval-quality, permission, finality or latency regressions.

Usage:
  uv run python benchmarks/eval_gate.py                       # BM25, compare to baseline
  uv run python benchmarks/eval_gate.py --retrieval hybrid    # needs local Ollama embeddings
  uv run python benchmarks/eval_gate.py --update-baseline     # accept current numbers

Corpus and labels: examples/eval/. Questions are split into `lexical` (reuse document
words), `paraphrase` (no shared key terms) and `restricted` (HR/finance documents).
Hard gates (always): 0 restricted chunks retrieved by an unauthorized user, and 0
post-revocation emissions in the TxnRAG race matrix. Soft gates: each quality metric may
drop at most TOLERANCE below the baseline; p95 retrieval latency must stay under budget.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from vaultkv import CacheProfile, Identity
from vaultkv.txnrag import Resource, TxnRAG
from vaultkv.txnrag.ingest import chunk_text
from vaultkv.txnrag.races import run_race_matrix
from vaultkv.txnrag.retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "examples" / "eval"
BASELINE = Path(__file__).resolve().parent / "eval_baseline.json"
TOLERANCE = 0.02
# Hybrid embeds each query with a CPU embedding model, so it gets a larger budget.
P95_BUDGET_MS = {"bm25": 250.0, "hybrid": 2000.0}
RESTRICTED = {"hr-compensation.md": "hana", "finance-forecast.md": "fin"}
PROFILE = CacheProfile("eval@1", "eval@1", "eval", "eval")


def build_store() -> TxnRAG:
    store = TxnRAG()
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted((EVAL / "corpus").glob("*.md"))}
    resources = {Resource("acme", name): chunk_text(text) for name, text in docs.items()}
    store.publish(resources, expected_epoch=0)
    for resource in resources:
        owner = RESTRICTED.get(resource.doc_id)
        users = [owner] if owner else ["alice", "hana", "fin"]
        store.set_permissions(
            resource,
            expected_epoch=store.resource_epoch(resource),
            principals=[Identity("acme", user) for user in users],
        )
    return store


def ranked_docs(store: TxnRAG, retriever: Retriever, user: str, question: str) -> list[str]:
    token = store.begin(Identity("acme", user), PROFILE)
    try:
        evidence = retriever.search(token, question, top_k=3)
        seen: list[str] = []
        for source in json.loads(evidence.payload):
            doc = source["source"]["resource"]["doc_id"]
            if doc not in seen:
                seen.append(doc)
        return seen
    finally:
        store.cancel(token)


def evaluate(retrieval: str) -> dict:
    store = build_store()
    if retrieval == "hybrid":
        from vaultkv.txnrag.ollama import OllamaClient

        client = OllamaClient()
        retriever = Retriever(store, client=client, model=client.resolve("nomic-embed-text:latest"))
    else:
        retriever = Retriever(store)
    questions = [json.loads(line) for line in (EVAL / "questions.jsonl").open()]
    by_kind: dict[str, list[tuple[int | None, float]]] = {}
    latencies, leaks = [], 0
    for q in questions:
        user = RESTRICTED.get(q["doc"], "alice")
        start = time.perf_counter()
        docs = ranked_docs(store, retriever, user, q["question"])
        latencies.append((time.perf_counter() - start) * 1000)
        rank = docs.index(q["doc"]) + 1 if q["doc"] in docs else None
        by_kind.setdefault(q["kind"], []).append(rank)
        if q["kind"] == "restricted":
            # The same question from a user without access must surface nothing restricted.
            leaks += sum(
                d in RESTRICTED for d in ranked_docs(store, retriever, "alice", q["question"])
            )
    store.close()

    def summarize(ranks: list[int | None]) -> dict:
        n = len(ranks)
        return {
            "n": n,
            "recall@1": round(sum(r == 1 for r in ranks) / n, 4),
            "recall@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 4),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
        }

    race = run_race_matrix()["post_revocation_emissions"]
    return {
        "retrieval": retrieval,
        "quality": {kind: summarize(ranks) for kind, ranks in sorted(by_kind.items())},
        "permission_leaks": leaks,
        "post_revocation_emissions": race["txnrag"],
        "p95_retrieval_ms": round(statistics.quantiles(latencies, n=20)[-1], 2),
    }


def check(report: dict, baseline: dict | None) -> list[str]:
    failures = []
    if report["permission_leaks"]:
        failures.append(f"permission leak: {report['permission_leaks']} restricted chunk(s)")
    if report["post_revocation_emissions"]:
        failures.append(
            f"finality: {report['post_revocation_emissions']} post-revocation emissions"
        )
    budget = P95_BUDGET_MS[report["retrieval"]]
    if report["p95_retrieval_ms"] > budget:
        failures.append(f"latency: p95 {report['p95_retrieval_ms']} ms > {budget} ms")
    for kind, metrics in (baseline or {}).get("quality", {}).items():
        current = report["quality"].get(kind, {})
        for name in ("recall@1", "recall@3", "mrr"):
            if current.get(name, 0) < metrics[name] - TOLERANCE:
                failures.append(f"{kind} {name}: {current.get(name)} < baseline {metrics[name]}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.retrieval)
    baselines = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    if args.update_baseline:
        baselines[args.retrieval] = {"quality": report["quality"]}
        BASELINE.write_text(json.dumps(baselines, indent=2) + "\n")
    failures = check(report, baselines.get(args.retrieval))
    print(json.dumps(report | {"failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
