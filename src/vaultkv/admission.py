"""Metadata-only admission checks: deny before probing cache contents."""

from __future__ import annotations

import sqlite3

from vaultkv.models import CacheProfile, Decision, Query, Scope, Status, Ticket, digest


def evaluate(db: sqlite3.Connection, query: Query, profile: CacheProfile) -> Decision:
    doc = db.execute(
        "SELECT * FROM documents WHERE tenant_ns = ? AND doc_id = ?",
        (query.owner_ns, query.doc_id),
    ).fetchone()
    if doc is None or doc["deleted"]:
        return Decision(Status.DENIED, "document_unavailable")
    scope = Scope(doc["acl_scope"])
    if scope != Scope.PUBLIC:
        if query.requester.tenant_ns != query.owner_ns:
            return Decision(Status.DENIED, "tenant_mismatch")
        allowed = db.execute(
            "SELECT 1 FROM permissions WHERE tenant_ns = ? AND doc_id = ? AND principal_id = ?",
            (query.owner_ns, query.doc_id, query.requester.principal_id),
        ).fetchone()
        if allowed is None:
            return Decision(Status.DENIED, "permission_denied")
    if doc["doc_version"] != query.doc_version:
        return Decision(Status.STALE, "version_mismatch")
    if doc["policy_epoch"] != query.policy_epoch:
        return Decision(Status.STALE, "epoch_mismatch")
    chunk = db.execute(
        "SELECT content_hash FROM chunks WHERE tenant_ns = ? AND doc_id = ? AND chunk_id = ?",
        (query.owner_ns, query.doc_id, query.chunk_id),
    ).fetchone()
    if chunk is None or chunk["content_hash"] != query.content_hash:
        return Decision(Status.STALE, "chunk_mismatch")
    key = cache_key(db, query, profile)
    cached = db.execute("SELECT 1 FROM cache_entries WHERE cache_key = ?", (key,)).fetchone()
    if cached is None:
        return Decision(Status.MISS, "not_cached")
    return Decision(Status.HIT, "current_and_authorized", Ticket(query, profile, key))


def cache_key(db: sqlite3.Connection, query: Query, profile: CacheProfile) -> str:
    """Call only after evaluate has authorized a hit/miss in the same transaction."""
    scope = db.execute(
        "SELECT acl_scope FROM documents WHERE tenant_ns = ? AND doc_id = ?",
        (query.owner_ns, query.doc_id),
    ).fetchone()[0]
    return digest(
        [
            query.owner_ns,
            query.doc_id,
            query.chunk_id,
            query.content_hash,
            query.doc_version,
            query.policy_epoch,
            scope,
            query.requester.principal_id if scope == Scope.PRIVATE else None,
            profile.fingerprint,
        ]
    )
