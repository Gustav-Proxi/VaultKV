"""Snapshot-compatible VaultKV materialization for trusted opaque byte producers."""

import hashlib
import sqlite3
from dataclasses import asdict, dataclass

from vaultkv.models import Provenance, Scope, digest
from vaultkv.txnrag.models import Artifact, ArtifactKind, TxnRejected, TxnToken
from vaultkv.txnrag.store import TxnRAG
from vaultkv.txnrag.telemetry import record_event


@dataclass(frozen=True)
class CacheTicket:
    context: Artifact
    cache_key: str


class SnapshotCache:
    """Whole-context byte-cache reference adapter; not position-independent tensor fusion."""

    def __init__(self, store: TxnRAG) -> None:
        self.store = store

    def _key(self, db: sqlite3.Connection, token: TxnToken, context: Artifact) -> str:
        self.store._check_artifact(db, token, context)
        if context.kind != ArtifactKind.CONTEXT:
            raise TxnRejected("not_a_context")
        dependencies = []
        for ref in context.dependencies:
            resource = self.store._resource(db, ref.resource)
            descriptor = asdict(ref)
            # Unchanged resources can be reused across K; versions/hashes must match.
            del descriptor["knowledge_epoch"]
            descriptor["scope"] = resource["scope"]
            descriptor["private_principal"] = (
                asdict(token.principal) if resource["scope"] == Scope.PRIVATE else None
            )
            dependencies.append(descriptor)
        return digest(
            {
                "sources": dependencies,
                "context": hashlib.sha256(context.payload).hexdigest(),
                "profile": token.profile.fingerprint,
                "policy_epoch": token.policy_epoch,
            }
        )

    def admit(self, token: TxnToken, context: Artifact) -> CacheTicket | None:
        with self.store._ledger.transaction() as db:
            self.store._guard(db, token)
            key = self._key(db, token, context)
            found = db.execute("SELECT 1 FROM txn_cache WHERE cache_key = ?", (key,)).fetchone()
            record_event(
                db, "synthetic_kv.lookup", {"key": key, "hit": bool(found)}, token.request_id
            )
            return CacheTicket(context, key) if found else None

    def publish(
        self, token: TxnToken, context: Artifact, provenance: Provenance, payload: bytes
    ) -> CacheTicket:
        if provenance.context_hash != token.profile.context_hash or provenance.position != 0:
            raise ValueError("Canonical position-zero provenance required")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("Cache payload must be nonempty bytes")
        with self.store._ledger.transaction() as db:
            self.store._guard(db, token)
            key = self._key(db, token, context)
            db.execute("INSERT OR IGNORE INTO txn_cache VALUES (?, ?)", (key, payload))
            record_event(
                db, "synthetic_kv.published", {"key": key, "bytes": len(payload)}, token.request_id
            )
            return CacheTicket(context, key)

    def load(self, token: TxnToken, ticket: CacheTicket) -> Artifact:
        """Recheck a queued lookup before materializing bytes inside trusted memory."""
        with self.store._ledger.transaction() as db:
            self.store._guard(db, token)
            key = self._key(db, token, ticket.context)
            if key != ticket.cache_key:
                raise TxnRejected("cache_ticket_mismatch")
            row = db.execute("SELECT payload FROM txn_cache WHERE cache_key = ?", (key,)).fetchone()
            if row is None:
                raise TxnRejected("cache_miss")
            record_event(
                db, "synthetic_kv.loaded", {"key": key, "bytes": len(row[0])}, token.request_id
            )
            return self.store._register(
                db, token, ArtifactKind.KV, ticket.context.dependencies, row[0]
            )


class AnswerCache:
    """Reuse deterministic buffered answers while preserving every authorization gate."""

    def __init__(self, store: TxnRAG) -> None:
        self.store = store

    def _key(self, db: sqlite3.Connection, token: TxnToken, context: Artifact) -> str:
        self.store._check_artifact(db, token, context)
        if context.kind != ArtifactKind.CONTEXT:
            raise TxnRejected("not_a_context")
        dependencies = []
        for ref in context.dependencies:
            resource = self.store._resource(db, ref.resource)
            descriptor = asdict(ref)
            del descriptor["knowledge_epoch"]
            descriptor["scope"] = resource["scope"]
            descriptor["private_principal"] = (
                asdict(token.principal) if resource["scope"] == Scope.PRIVATE else None
            )
            dependencies.append(descriptor)
        return digest(
            {
                "kind": "buffered-answer-v1",
                "sources": dependencies,
                "context": hashlib.sha256(context.payload).hexdigest(),
                "profile": token.profile.fingerprint,
                "policy_epoch": token.policy_epoch,
            }
        )

    def get(self, token: TxnToken, context: Artifact) -> bytes | None:
        with self.store._ledger.transaction() as db:
            self.store._guard(db, token)
            key = self._key(db, token, context)
            row = db.execute(
                "SELECT payload FROM txn_answers WHERE cache_key = ?", (key,)
            ).fetchone()
            record_event(db, "answer.cache", {"hit": row is not None}, token.request_id)
            return row[0] if row else None

    def put(self, token: TxnToken, context: Artifact, payload: bytes) -> None:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("Cached answer must be nonempty bytes")
        with self.store._ledger.transaction() as db:
            self.store._guard(db, token)
            key = self._key(db, token, context)
            db.execute("INSERT OR IGNORE INTO txn_answers VALUES (?, ?)", (key, payload))
