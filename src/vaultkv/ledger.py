"""SQLite metadata ledger and transactional invalidation.

All state transitions and guarded consumers acquire the same database write lock.
This intentionally favors simple local correctness over serving throughput.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from vaultkv.models import (
    Chunk,
    ConflictError,
    Document,
    Identity,
    Scope,
    content_hash,
    require_name,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_version INTEGER NOT NULL CHECK (doc_version > 0),
    policy_epoch INTEGER NOT NULL CHECK (policy_epoch > 0),
    acl_scope TEXT NOT NULL CHECK (acl_scope IN ('public', 'tenant-shared', 'private')),
    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
    PRIMARY KEY (tenant_ns, doc_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (tenant_ns, doc_id, chunk_id),
    FOREIGN KEY (tenant_ns, doc_id) REFERENCES documents (tenant_ns, doc_id)
);
CREATE TABLE IF NOT EXISTS permissions (
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    PRIMARY KEY (tenant_ns, doc_id, principal_id),
    FOREIGN KEY (tenant_ns, doc_id) REFERENCES documents (tenant_ns, doc_id)
);
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key TEXT PRIMARY KEY,
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    payload BLOB NOT NULL,
    FOREIGN KEY (tenant_ns, doc_id) REFERENCES documents (tenant_ns, doc_id)
);
CREATE INDEX IF NOT EXISTS cache_by_document ON cache_entries (tenant_ns, doc_id);
"""


class Ledger:
    """Trusted administrative API, not an authorization boundary for clients."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = RLock()
        self._db = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False, timeout=5.0
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._db.in_transaction:
                raise RuntimeError("Cannot close ledger during a guarded operation")
            self._db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Internal boundary shared by ledger, cache, and admission components."""
        with self._lock:
            if self._db.in_transaction:
                raise RuntimeError("Reentrant ledger operations are not allowed")
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    @staticmethod
    def _document(db: sqlite3.Connection, tenant_ns: str, doc_id: str) -> Document:
        row = db.execute(
            "SELECT * FROM documents WHERE tenant_ns = ? AND doc_id = ?", (tenant_ns, doc_id)
        ).fetchone()
        if row is None:
            raise KeyError((tenant_ns, doc_id))
        chunks = db.execute(
            "SELECT chunk_id, content_hash FROM chunks "
            "WHERE tenant_ns = ? AND doc_id = ? ORDER BY chunk_id",
            (tenant_ns, doc_id),
        ).fetchall()
        principals = db.execute(
            "SELECT principal_id FROM permissions WHERE tenant_ns = ? AND doc_id = ?",
            (tenant_ns, doc_id),
        ).fetchall()
        return Document(
            tenant_ns,
            doc_id,
            row["doc_version"],
            row["policy_epoch"],
            Scope(row["acl_scope"]),
            bool(row["deleted"]),
            tuple(Chunk(*chunk) for chunk in chunks),
            frozenset(Identity(tenant_ns, item[0]) for item in principals),
        )

    def snapshot(self, tenant_ns: str, doc_id: str) -> Document:
        """Read administrative metadata, including tombstones; no chunk text is stored."""
        with self.transaction() as db:
            return self._document(db, tenant_ns, doc_id)

    @staticmethod
    def _validate_policy(
        tenant_ns: str, acl_scope: Scope, principals: Iterable[Identity]
    ) -> frozenset[Identity]:
        Scope(acl_scope)
        members = frozenset(principals)
        if any(member.tenant_ns != tenant_ns for member in members):
            raise ValueError("Allowed users must belong to the document's workspace")
        if acl_scope == Scope.PUBLIC and members:
            raise ValueError("Public documents do not use an allowed-user list")
        return members

    @staticmethod
    def _invalidate(db: sqlite3.Connection, tenant_ns: str, doc_id: str) -> None:
        db.execute(
            "DELETE FROM cache_entries WHERE tenant_ns = ? AND doc_id = ?", (tenant_ns, doc_id)
        )

    @staticmethod
    def _permissions(
        db: sqlite3.Connection, tenant_ns: str, doc_id: str, members: frozenset[Identity]
    ) -> None:
        db.execute(
            "DELETE FROM permissions WHERE tenant_ns = ? AND doc_id = ?", (tenant_ns, doc_id)
        )
        db.executemany(
            "INSERT INTO permissions VALUES (?, ?, ?)",
            [(tenant_ns, doc_id, member.principal_id) for member in members],
        )

    def put_document(
        self,
        tenant_ns: str,
        doc_id: str,
        chunks: Mapping[str, str],
        *,
        expected_version: int,
        expected_epoch: int | None = None,
        acl_scope: Scope = Scope.TENANT,
        principals: Iterable[Identity] = (),
    ) -> Document:
        """Replace a document and its ACL atomically. Creation expects version zero.

        The caller supplies a complete snapshot, including permissions on every edit.
        Existing documents require both their current version and policy epoch.
        Versions survive deletion, preventing old tickets from matching recreated docs.
        """
        require_name(tenant_ns)
        require_name(doc_id)
        members = self._validate_policy(tenant_ns, acl_scope, principals)
        chunk_rows = []
        for chunk_id, text in chunks.items():
            require_name(chunk_id)
            chunk_rows.append((tenant_ns, doc_id, chunk_id, content_hash(text)))
        with self.transaction() as db:
            try:
                old = self._document(db, tenant_ns, doc_id)
            except KeyError:
                old = None
            if expected_version != (old.doc_version if old else 0):
                raise ConflictError("Document version changed; refresh before writing")
            if old and expected_epoch != old.policy_epoch:
                raise ConflictError("Policy epoch changed or missing; refresh before writing")
            version = expected_version + 1
            epoch = 1 if old is None else old.policy_epoch
            if old and (old.acl_scope != acl_scope or old.principals != members or old.deleted):
                epoch += 1
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(tenant_ns, doc_id) DO UPDATE SET "
                "doc_version = excluded.doc_version, policy_epoch = excluded.policy_epoch, "
                "acl_scope = excluded.acl_scope, deleted = 0",
                (tenant_ns, doc_id, version, epoch, acl_scope),
            )
            self._invalidate(db, tenant_ns, doc_id)
            db.execute("DELETE FROM chunks WHERE tenant_ns = ? AND doc_id = ?", (tenant_ns, doc_id))
            db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", chunk_rows)
            self._permissions(db, tenant_ns, doc_id, members)
            return self._document(db, tenant_ns, doc_id)

    def set_permissions(
        self,
        tenant_ns: str,
        doc_id: str,
        *,
        expected_epoch: int,
        acl_scope: Scope,
        principals: Iterable[Identity] = (),
    ) -> Document:
        members = self._validate_policy(tenant_ns, acl_scope, principals)
        with self.transaction() as db:
            old = self._document(db, tenant_ns, doc_id)
            if old.deleted:
                raise ValueError("Cannot change permissions on a deleted document")
            if old.policy_epoch != expected_epoch:
                raise ConflictError("Policy epoch changed; refresh before writing")
            if old.acl_scope != acl_scope or old.principals != members:
                db.execute(
                    "UPDATE documents SET policy_epoch = policy_epoch + 1, acl_scope = ? "
                    "WHERE tenant_ns = ? AND doc_id = ?",
                    (acl_scope, tenant_ns, doc_id),
                )
                self._permissions(db, tenant_ns, doc_id, members)
                self._invalidate(db, tenant_ns, doc_id)
            return self._document(db, tenant_ns, doc_id)

    def delete_document(self, tenant_ns: str, doc_id: str, *, expected_version: int) -> Document:
        with self.transaction() as db:
            old = self._document(db, tenant_ns, doc_id)
            if old.doc_version != expected_version:
                raise ConflictError("Document version changed; refresh before deleting")
            if not old.deleted:
                db.execute(
                    "UPDATE documents SET deleted = 1, doc_version = doc_version + 1, "
                    "policy_epoch = policy_epoch + 1 WHERE tenant_ns = ? AND doc_id = ?",
                    (tenant_ns, doc_id),
                )
                self._invalidate(db, tenant_ns, doc_id)
                db.execute(
                    "DELETE FROM chunks WHERE tenant_ns = ? AND doc_id = ?", (tenant_ns, doc_id)
                )
                self._permissions(db, tenant_ns, doc_id, frozenset())
            return self._document(db, tenant_ns, doc_id)
