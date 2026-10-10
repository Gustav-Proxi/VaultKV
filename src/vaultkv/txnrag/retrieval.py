"""ACL-filtered BM25 and optional dense retrieval over a pinned knowledge snapshot."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass

from vaultkv.models import digest
from vaultkv.txnrag.models import (
    Artifact,
    ArtifactKind,
    EvidenceRef,
    Resource,
    TxnRejected,
    TxnToken,
)
from vaultkv.txnrag.ollama import ModelError, ModelRef, OllamaClient
from vaultkv.txnrag.store import TxnRAG, _json


@dataclass(frozen=True)
class _Candidate:
    ref: EvidenceRef
    text: str


def bm25(query: str, documents: list[str]) -> list[float]:
    """Positive-IDF BM25, k1=1.2/b=0.75, Unicode word tokenization."""
    tokens = [Counter(re.findall(r"\w+", text.casefold())) for text in documents]
    terms = set(re.findall(r"\w+", query.casefold()))
    average = sum(sum(row.values()) for row in tokens) / len(tokens) if tokens else 0
    if not average:
        return [0.0] * len(documents)
    frequency = Counter(term for row in tokens for term in row)
    scores = []
    for row in tokens:
        score, length = 0.0, sum(row.values())
        for term in terms:
            count = row[term]
            if count:
                idf = math.log(1 + (len(tokens) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / average))
        scores.append(score)
    return scores


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ModelError("Embedding dimensions changed; rebuild with a fixed model revision")
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    if denominator == 0 or not math.isfinite(denominator):
        raise ModelError("Invalid embedding norm")
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator


class Retriever:
    def __init__(
        self,
        store: TxnRAG,
        *,
        client: OllamaClient | None = None,
        model: ModelRef | None = None,
        adaptive: bool = False,
        observe: Callable[[str, dict], None] = lambda kind, details: None,
    ) -> None:
        if (client is None) != (model is None):
            raise ValueError("Dense retrieval needs both an Ollama client and embedding model")
        self.store, self.client, self.model = store, client, model
        self.adaptive = adaptive
        self.observe = observe

    def _candidates(self, token: TxnToken) -> list[_Candidate]:
        candidates = []
        with self.store._ledger.transaction() as db:
            self.store._guard(db, token)
            resources = db.execute(
                "SELECT * FROM txn_snapshot_docs WHERE epoch = ? ORDER BY tenant_ns, doc_id",
                (token.knowledge_epoch,),
            ).fetchall()
            for row in resources:
                resource = Resource(row["tenant_ns"], row["doc_id"])
                try:
                    authorization = self.store._authorize(db, token, resource)
                except TxnRejected as exc:
                    if exc.reason == "permission_denied":
                        continue
                    raise
                # No unauthorized text is loaded or sent to the embedding service.
                for chunk in db.execute(
                    "SELECT * FROM txn_chunks WHERE tenant_ns = ? AND doc_id = ? "
                    "AND doc_version = ? ORDER BY chunk_id",
                    (resource.tenant_ns, resource.doc_id, row["doc_version"]),
                ):
                    ref = EvidenceRef(
                        resource,
                        row["doc_version"],
                        chunk["chunk_id"],
                        chunk["content_hash"],
                        token.knowledge_epoch,
                        authorization["auth_epoch"],
                    )
                    candidates.append(_Candidate(ref, chunk["text"]))
        return candidates

    def _check_candidates(self, db, token: TxnToken, candidates: list[_Candidate]) -> None:
        self.store._guard(db, token)
        for item in candidates:
            self.store._validate_ref(db, token, item.ref)
            row = self.store._authorize(db, token, item.ref.resource)
            if row["auth_epoch"] != item.ref.resource_auth_epoch:
                raise TxnRejected("resource_authorization_changed")

    def _dense(self, token: TxnToken, query: str, candidates: list[_Candidate]) -> list[float]:
        assert self.client is not None and self.model is not None
        keys = []
        vectors = []
        missing = []
        query_key = digest(["query", query, self.model.revision, "nomic-search-prefixes-v1"])
        with self.store._ledger.transaction() as db:
            self._check_candidates(db, token, candidates)
            for index, item in enumerate(candidates):
                ref = asdict(item.ref)
                del ref["knowledge_epoch"]
                del ref["resource_auth_epoch"]
                # Embeddings depend on content, source identity, producer, and preprocessing.
                key = digest([ref, self.model.revision, "nomic-search-prefixes-v1"])
                keys.append(key)
                row = db.execute(
                    "SELECT vector_json FROM txn_embeddings WHERE cache_key = ?", (key,)
                ).fetchone()
                vectors.append(json.loads(row[0]) if row else None)
                if row is None:
                    missing.append(index)
            row = db.execute(
                "SELECT vector_json FROM txn_embeddings WHERE cache_key = ?", (query_key,)
            ).fetchone()
            query_vector = json.loads(row[0]) if row else None
        self.observe(
            "embedding.cache",
            {
                "hits": len(candidates) - len(missing),
                "misses": len(missing),
                "query_hit": query_vector is not None,
                "model": self.model.revision,
            },
        )
        # Expensive model calls run outside the authorization lock. Persist each batch as
        # soon as it is produced, so a large corpus survives timeouts and concurrent
        # requests reuse progress instead of re-embedding everything.
        for start in range(0, len(missing), 16):
            batch = missing[start : start + 16]
            produced = self.client.embed(
                self.model, ["search_document: " + candidates[i].text for i in batch]
            )
            for index, vector in zip(batch, produced, strict=True):
                vectors[index] = vector
            with self.store._ledger.transaction() as db:
                self._check_candidates(db, token, [candidates[i] for i in batch])
                db.executemany(
                    "INSERT OR IGNORE INTO txn_embeddings VALUES (?, ?)",
                    [(keys[index], _json(vectors[index])) for index in batch],
                )
        if query_vector is None:
            query_vector = self.client.embed(self.model, ["search_query: " + query])[0]
        self.observe(
            "embedding.completed",
            {
                "dimensions": len(query_vector),
                "generated_vectors": len(missing),
                "query_vectors": 1,
                "query_cache_hit": row is not None,
            },
        )
        with self.store._ledger.transaction() as db:
            self._check_candidates(db, token, candidates)
            db.executemany(
                "INSERT OR IGNORE INTO txn_embeddings VALUES (?, ?)",
                [(keys[index], _json(vectors[index])) for index in missing],
            )
            db.execute(
                "INSERT OR IGNORE INTO txn_embeddings VALUES (?, ?)",
                (query_key, _json(query_vector)),
            )
        return [cosine(query_vector, vector) for vector in vectors]

    def search(self, token: TxnToken, query: str, *, top_k: int = 3) -> Artifact:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        candidates = self._candidates(token)
        self.observe("retrieval.candidates", {"authorized_chunks": len(candidates)})
        lexical = bm25(query, [item.text for item in candidates])
        sparse_order = sorted(
            (i for i, score in enumerate(lexical) if score > 0), key=lambda i: (-lexical[i], i)
        )
        scores = {index: 1 / (60 + rank) for rank, index in enumerate(sparse_order, 1)}
        dense = None
        use_dense = self.client is not None and bool(candidates)
        if use_dense and self.adaptive and sparse_order:
            first = sparse_order[0]
            runner_up = lexical[sparse_order[1]] if len(sparse_order) > 1 else 0.0
            terms = set(re.findall(r"\w+", query.casefold()))
            document_terms = set(re.findall(r"\w+", candidates[first].text.casefold()))
            coverage = len(terms & document_terms) / len(terms) if terms else 0.0
            decisive = coverage >= 0.25 and (runner_up == 0 or lexical[first] >= runner_up * 2)
            use_dense = not decisive
            if decisive:
                # Keep a direct answer small instead of including weak stop-word matches.
                scores = {first: scores[first]}
            self.observe(
                "retrieval.strategy",
                {
                    "dense_used": use_dense,
                    "reason": "meaning_fallback" if use_dense else "clear_keyword_match",
                },
            )
        if use_dense:
            dense = self._dense(token, query, candidates)
            dense_order = sorted(
                (i for i, score in enumerate(dense) if score > 0), key=lambda i: (-dense[i], i)
            )
            for rank, index in enumerate(dense_order, 1):
                scores[index] = scores.get(index, 0) + 1 / (60 + rank)
        selected = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        self.observe(
            "retrieval.ranked",
            {
                "candidates": [
                    {
                        "resource": asdict(item.ref.resource),
                        "chunk_id": item.ref.chunk_id,
                        "doc_version": item.ref.doc_version,
                        "content_hash": item.ref.content_hash,
                        "bm25": lexical[i],
                        "cosine": dense[i] if dense is not None else None,
                        "rrf": scores.get(i, 0),
                        "selected": i in selected,
                    }
                    for i, item in enumerate(candidates)
                ]
            },
        )
        with self.store._ledger.transaction() as db:
            # Fail if any ranking input lost authorization during external computation.
            self._check_candidates(db, token, candidates)
            refs, payload = [], []
            for number, index in enumerate(selected, 1):
                item = candidates[index]
                refs.append(item.ref)
                payload.append(
                    {
                        "citation": f"S{number}",
                        "source": asdict(item.ref),
                        "text": item.text,
                        "score": scores[index],
                        "bm25": lexical[index],
                        "cosine": dense[index] if dense is not None else None,
                    }
                )
                db.execute(
                    "INSERT OR IGNORE INTO txn_dependencies VALUES (?, ?)",
                    (token.request_id, _json(asdict(item.ref))),
                )
            return self.store._register(
                db, token, ArtifactKind.EVIDENCE, refs, _json(payload).encode()
            )
