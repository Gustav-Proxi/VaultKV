"""Transactional metadata events. Never record prompts, chunks, or raw model output."""

import json
import sqlite3
import time


def record_event(
    db: sqlite3.Connection, kind: str, details: dict, request_id: str | None = None
) -> None:
    db.execute(
        "INSERT INTO txn_events(created_at, request_id, kind, details_json) VALUES (?, ?, ?, ?)",
        (time.time(), request_id, kind, json.dumps(details, sort_keys=True)),
    )


class RequestTrace:
    def __init__(self, store, token, question: str) -> None:
        self.store, self.token = store, token
        self.started = time.perf_counter()
        with store._ledger.transaction() as db:
            db.execute(
                "INSERT INTO txn_traces(request_id, created_at, question, status) "
                "VALUES (?, ?, ?, 'running')",
                (token.request_id, time.time(), question),
            )
            record_event(
                db,
                "request.started",
                {
                    "knowledge_epoch": token.knowledge_epoch,
                    "policy_epoch": token.policy_epoch,
                    "principal_auth_epoch": token.principal_auth_epoch,
                },
                token.request_id,
            )

    def event(self, kind: str, details: dict | None = None) -> None:
        with self.store._ledger.transaction() as db:
            record_event(
                db,
                kind,
                {
                    **(details or {}),
                    "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3),
                },
                self.token.request_id,
            )

    def released(self, payload: bytes) -> None:
        """Called ONLY inside release's transaction, after its authorization check."""
        db = self.store._ledger._db
        if not db.in_transaction:
            raise RuntimeError("Trace output must be recorded inside the finality gate")
        elapsed = (time.perf_counter() - self.started) * 1000
        db.execute(
            "UPDATE txn_traces SET status = 'emitted', result_json = ?, finished_at = ?, "
            "elapsed_ms = ? WHERE request_id = ?",
            (payload.decode(), time.time(), elapsed, self.token.request_id),
        )
        record_event(db, "finality.accepted", {"elapsed_ms": elapsed}, self.token.request_id)

    def failed(self, error: Exception) -> None:
        from vaultkv.txnrag.models import TxnRejected

        reason = error.reason if isinstance(error, TxnRejected) else type(error).__name__
        status = "blocked" if isinstance(error, TxnRejected) else "failed"
        elapsed = (time.perf_counter() - self.started) * 1000
        with self.store._ledger.transaction() as db:
            db.execute(
                "UPDATE txn_traces SET status = ?, error = ?, result_json = NULL, "
                "finished_at = ?, elapsed_ms = ? WHERE request_id = ?",
                (status, reason, time.time(), elapsed, self.token.request_id),
            )
            record_event(
                db,
                "request." + status,
                {
                    "reason": reason,
                    "elapsed_ms": elapsed,
                },
                self.token.request_id,
            )
