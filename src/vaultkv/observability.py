"""Operator inspection with current authorization on every historical content read."""

from __future__ import annotations

import inspect
import json
import statistics
from collections.abc import Callable

from vaultkv.connectors.runtime import doctor
from vaultkv.models import CacheProfile, Identity
from vaultkv.txnrag import ReadMode, TxnRAG, TxnRejected, TxnToken


def _token(raw: str) -> TxnToken:
    data = json.loads(raw)
    data["principal"] = Identity(**data["principal"])
    data["profile"] = CacheProfile(**data["profile"])
    data["mode"] = ReadMode(data["mode"])
    return TxnToken(**data)


def _event(row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "request_id": row["request_id"],
        "kind": row["kind"],
        "details": json.loads(row["details_json"]),
    }


def _summary(row) -> dict:
    token = json.loads(row["token_json"])
    return {
        key: row[key]
        for key in (
            "request_id",
            "created_at",
            "finished_at",
            "elapsed_ms",
            "question",
            "status",
            "error",
        )
    } | {
        "principal": token["principal"],
        "knowledge_epoch": token["knowledge_epoch"],
        "model_revision": token["profile"]["model_revision"],
    }


def list_traces(store: TxnRAG, *, limit: int = 100) -> list[dict]:
    with store._ledger.transaction() as db:
        rows = db.execute(
            "SELECT t.*, r.token_json FROM txn_traces t "
            "JOIN txn_requests r USING(request_id) "
            "ORDER BY t.created_at DESC LIMIT ?",
            (min(max(limit, 1), 500),),
        ).fetchall()
        return [_summary(row) for row in rows]


def snapshot(store: TxnRAG) -> dict:
    with store._ledger.transaction() as db:
        meta = db.execute("SELECT * FROM txn_meta").fetchone()
        counts = dict(db.execute("SELECT status, count(*) FROM txn_traces GROUP BY status"))
        durations = [
            row[0]
            for row in db.execute(
                "SELECT elapsed_ms FROM txn_traces WHERE status = 'emitted' "
                "AND elapsed_ms IS NOT NULL"
            )
        ]
        documents = []
        for row in db.execute(
            "SELECT s.*, r.auth_epoch, r.scope FROM txn_snapshot_docs s "
            "JOIN txn_resources r USING(tenant_ns, doc_id) WHERE s.epoch = ? "
            "ORDER BY s.tenant_ns, s.doc_id",
            (meta["knowledge_epoch"],),
        ).fetchall():
            doc = dict(row)
            doc["principals"] = [
                item[0]
                for item in db.execute(
                    "SELECT principal_id FROM txn_acl WHERE tenant_ns = ? AND doc_id = ? "
                    "ORDER BY principal_id",
                    (row["tenant_ns"], row["doc_id"]),
                )
            ]
            doc["groups"] = [
                item[0]
                for item in db.execute(
                    "SELECT group_id FROM txn_acl_groups WHERE tenant_ns = ? AND doc_id = ? "
                    "ORDER BY group_id",
                    (row["tenant_ns"], row["doc_id"]),
                )
            ]
            doc["chunks"] = db.execute(
                "SELECT count(*) FROM txn_chunks WHERE tenant_ns = ? AND doc_id = ? "
                "AND doc_version = ?",
                (row["tenant_ns"], row["doc_id"], row["doc_version"]),
            ).fetchone()[0]
            documents.append(doc)
        cache = db.execute(
            "SELECT count(*), coalesce(sum(length(payload)), 0) FROM txn_cache"
        ).fetchone()
        embedding = db.execute("SELECT count(*) FROM txn_embeddings").fetchone()[0]
        answers = db.execute("SELECT count(*) FROM txn_answers").fetchone()[0]
        events = [
            _event(row) for row in db.execute("SELECT * FROM txn_events ORDER BY id DESC LIMIT 80")
        ]
        transfers = [
            json.loads(row[0])["tokens"]
            for row in db.execute(
                "SELECT details_json FROM txn_events WHERE kind = 'connector.loaded'"
            )
        ]
        return {
            "connector": doctor(),
            "knowledge_epoch": meta["knowledge_epoch"],
            "policy_epoch": meta["policy_epoch"],
            "request_counts": counts,
            "p50_ms": statistics.median(durations) if durations else None,
            "documents": documents,
            "embedding_entries": embedding,
            "answer_entries": answers,
            "synthetic_kv": {"entries": cache[0], "bytes": cache[1]},
            "tensor_kv": {
                "status": "transfers_recorded" if transfers else "not_connected",
                "loaded_tokens": sum(transfers) if transfers else None,
                "load_events": len(transfers),
                "hit_rate": None,
                "bytes": None,
            },
            "events": events,
        }


def inspect_trace(store: TxnRAG, request_id: str, sink: Callable[[dict], object]) -> object:
    """Hold the authorization lock through output, including subsequent dashboard reads."""
    with store._ledger.transaction() as db:
        row = db.execute(
            "SELECT t.*, r.token_json, r.state FROM txn_traces t "
            "JOIN txn_requests r USING(request_id) "
            "WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        token = _token(row["token_json"])
        result, withheld = None, None
        if row["status"] == "emitted" and row["state"] == "emitted" and row["result_json"]:
            try:
                store._check_identity(db, token)
                for ref in store._dependencies(db, token):
                    store._validate_ref(db, token, ref)
                    current = store._authorize(db, token, ref.resource)
                    if current["auth_epoch"] != ref.resource_auth_epoch:
                        raise TxnRejected("resource_authorization_changed")
                result = json.loads(row["result_json"])
            except TxnRejected as exc:
                withheld = exc.reason
        detail = _summary(row) | {
            "contract": json.loads(row["token_json"]),
            "result": result,
            "withheld": withheld,
            "events": [
                _event(event)
                for event in db.execute(
                    "SELECT * FROM txn_events WHERE request_id = ? ORDER BY id",
                    (request_id,),
                )
            ],
        }
        result = sink(detail)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("Inspection sinks must be synchronous")
        return result


def vectorlens_session(trace: dict) -> dict:
    """Adapter for ~/Code/vectorlens/vectorlens/types.py; no auto-interception.

    Attribution arrays stay empty: retrieval scores are not causal influence or
    hallucination measurements. Only currently releasable content is exported.
    """
    identifier = trace["request_id"]
    result = trace["result"]
    generation = next(
        (e["details"] for e in trace["events"] if e["kind"] == "generation.completed"), {}
    )
    session = {
        "id": identifier,
        "created_at": trace["created_at"],
        "conversation_id": identifier,
        "vector_queries": [],
        "llm_requests": [],
        "llm_responses": [],
        "attributions": [],
    }
    if result is None:
        return session
    session["vector_queries"] = [
        {
            "id": identifier + ":retrieval",
            "session_id": identifier,
            "timestamp": trace["created_at"],
            "db_type": "txnrag-sqlite-" + result["retrieval"],
            "collection": f"snapshot-{trace['knowledge_epoch']}",
            "query_text": trace["question"],
            "query_embedding": [],
            "top_k": len(result["sources"]),
            "results": [
                {
                    "chunk_id": (
                        f"{s['source']['resource']['tenant_ns']}/"
                        f"{s['source']['resource']['doc_id']}:{s['source']['chunk_id']}"
                    ),
                    "text": s["excerpt"],
                    "score": s.get("scores", {}).get("rrf", 0),
                    "metadata": {
                        **s["source"],
                        "citation": s["id"],
                        "scores": s.get("scores", {}),
                        "attribution_measured": False,
                    },
                }
                for s in result["sources"]
            ],
        }
    ]
    session["llm_requests"] = [
        {
            "id": identifier + ":generation",
            "session_id": identifier,
            "timestamp": trace["created_at"],
            "provider": result.get("generation_backend", "ollama"),
            "model": result["model"],
            "system_prompt": "",
            "messages": [{"role": "user", "content": trace["question"]}],
            "temperature": 0,
            "max_tokens": 384,
            "vector_query_id": identifier + ":retrieval",
            "chain_step": "generation",
        }
    ]
    session["llm_responses"] = [
        {
            "id": identifier + ":response",
            "session_id": identifier,
            "request_id": identifier + ":generation",
            "timestamp": trace["finished_at"],
            "output_text": result["answer"],
            "output_tokens": [],
            "latency_ms": generation.get("total_duration", 0) / 1e6,
            "prompt_tokens": generation.get("prompt_eval_count", 0),
            "completion_tokens": generation.get("eval_count", 0),
            "cost_usd": 0,
        }
    ]
    return session
