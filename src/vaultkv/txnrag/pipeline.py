"""Deterministic CPU reference stages. No language model or learned reranker."""

import json
from collections.abc import Callable, Iterable
from enum import StrEnum

from vaultkv import Provenance
from vaultkv.txnrag import Artifact, ArtifactKind, Resource, SnapshotCache, TxnRAG, TxnToken


class Phase(StrEnum):
    BEFORE_RETRIEVAL = "before_retrieval"
    AFTER_RETRIEVAL = "after_retrieval"
    AFTER_RERANKING = "after_reranking"
    AFTER_CONTEXT = "after_context"
    AFTER_KV_LOOKUP = "after_kv_lookup"
    AFTER_KV_LOAD = "after_kv_load"
    DURING_GENERATION = "during_generation"
    BEFORE_RELEASE = "before_release"


def prepare_answer(
    store: TxnRAG,
    token: TxnToken,
    resources: Iterable[Resource],
    *,
    barrier: Callable[[Phase], None] = lambda phase: None,
) -> Artifact:
    """Generate a private extractive buffer; the caller must release it through the gate."""
    barrier(Phase.BEFORE_RETRIEVAL)
    evidence = store.retrieve(token, resources)
    barrier(Phase.AFTER_RETRIEVAL)
    # Identity rerank retains the complete source closure for this reference pipeline.
    ranked = store.derive(token, ArtifactKind.RERANKED, [evidence], evidence.payload)
    barrier(Phase.AFTER_RERANKING)
    text = "\n".join(item["text"] for item in json.loads(ranked.payload))
    context = store.derive(token, ArtifactKind.CONTEXT, [ranked], text.encode())
    barrier(Phase.AFTER_CONTEXT)
    cache = SnapshotCache(store)
    ticket = cache.admit(token, context)
    if ticket is None:
        ticket = cache.publish(
            token,
            context,
            Provenance(token.profile.context_hash),
            b"synthetic-kv:" + context.payload,
        )
    barrier(Phase.AFTER_KV_LOOKUP)
    kv = cache.load(token, ticket)
    barrier(Phase.AFTER_KV_LOAD)
    barrier(Phase.DURING_GENERATION)
    # CPU extractive stand-in: generation occurs with NO database lock held.
    answer = store.derive(token, ArtifactKind.ANSWER, [context, kv], context.payload)
    barrier(Phase.BEFORE_RELEASE)
    return answer
