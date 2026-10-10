"""Controlled ablations, including deliberately unsafe benchmark-only baselines."""

import json

from vaultkv import CacheProfile, Identity, Provenance
from vaultkv.txnrag import ArtifactKind, Resource, SnapshotCache, TxnRAG, TxnRejected
from vaultkv.txnrag.pipeline import Phase, prepare_answer

PROFILE = CacheProfile("synthetic-model@1", "synthetic-tokenizer@1", "bytes-v1", "canonical")
PRINCIPAL = Identity("acme", "alice")
RESOURCE = Resource("acme", "confidential")
SECRET = b"CONFIDENTIAL: project budget is 120000"


def _seed(store: TxnRAG) -> None:
    store.publish({RESOURCE: {"budget": SECRET.decode()}}, expected_epoch=0)
    store.set_permissions(RESOURCE, expected_epoch=1, principals=[PRINCIPAL])


def run_case(strategy: str, revoke_at: Phase) -> dict[str, object]:
    """Ground truth is the fault injector's commit flag, not ledger/admission reads."""
    if strategy not in ("retrieval_only", "retrieval_and_kv", "txnrag"):
        raise ValueError("Unknown baseline")
    revoked, emissions, leaks = False, 0, 0
    rejection = None
    with TxnRAG() as store:
        _seed(store)
        token = store.begin(PRINCIPAL, PROFILE)

        def barrier(phase: Phase) -> None:
            nonlocal revoked
            if phase == revoke_at:
                store.set_permissions(RESOURCE, expected_epoch=2)
                revoked = True  # Mutation returned only after its transaction committed.

        def sink(payload: bytes) -> None:
            nonlocal emissions, leaks
            emissions += 1
            if revoked and SECRET in payload:
                leaks += 1

        try:
            if strategy == "txnrag":
                answer = prepare_answer(store, token, [RESOURCE], barrier=barrier)
                store.release(token, answer, sink)
            else:
                # Keep unsafe baseline bypasses confined to the experiment module.
                barrier(Phase.BEFORE_RETRIEVAL)
                evidence = store.retrieve(token, [RESOURCE])
                barrier(Phase.AFTER_RETRIEVAL)
                ranked = store.derive(token, ArtifactKind.RERANKED, [evidence], evidence.payload)
                barrier(Phase.AFTER_RERANKING)
                text = "\n".join(item["text"] for item in json.loads(ranked.payload)).encode()
                context = store.derive(token, ArtifactKind.CONTEXT, [ranked], text)
                barrier(Phase.AFTER_CONTEXT)
                if strategy == "retrieval_and_kv":
                    cache = SnapshotCache(store)
                    ticket = cache.publish(token, context, Provenance(PROFILE.context_hash), b"kv")
                barrier(Phase.AFTER_KV_LOOKUP)
                if strategy == "retrieval_and_kv":
                    cache.load(token, ticket)
                barrier(Phase.AFTER_KV_LOAD)
                barrier(Phase.DURING_GENERATION)
                barrier(Phase.BEFORE_RELEASE)
                sink(text)  # Deliberate benchmark-only omission of the finality gate.
        except TxnRejected as exc:
            rejection = exc.reason
        finally:
            store.cancel(token)
    return {
        "strategy": strategy,
        "revoke_at": revoke_at.value,
        "emissions": emissions,
        "post_revocation_emissions": leaks,
        "rejected": rejection,
    }


def run_race_matrix() -> dict[str, object]:
    strategies = ("retrieval_only", "retrieval_and_kv", "txnrag")
    cases = [run_case(strategy, phase) for strategy in strategies for phase in Phase]
    totals = {
        strategy: sum(
            row["post_revocation_emissions"] for row in cases if row["strategy"] == strategy
        )
        for strategy in strategies
    }
    if totals["txnrag"] != 0:
        raise AssertionError("TxnRAG emitted protected output after revocation")
    if totals["retrieval_only"] == 0 or totals["retrieval_and_kv"] == 0:
        raise AssertionError("Unsafe baselines did not exercise the expected races")
    return {
        "mode": "synthetic-cpu-race-matrix",
        "cases": cases,
        "post_revocation_emissions": totals,
    }


def demo() -> dict[str, object]:
    with TxnRAG() as store:
        _seed(store)
        token = store.begin(PRINCIPAL, PROFILE)
        answer = prepare_answer(store, token, [RESOURCE])
        store.publish({RESOURCE: {"budget": "Revised budget is 150000"}}, expected_epoch=1)
        output = []
        store.release(token, answer, lambda payload: output.append(payload.decode()))
        snapshot_result = {"request_epoch": 1, "current_epoch": 2, "output": output[0]}
    return {
        "mode": "synthetic-cpu-txnrag-demo",
        "pinned_snapshot": snapshot_result,
        "mid_generation_revocation": run_case("txnrag", Phase.DURING_GENERATION),
    }
