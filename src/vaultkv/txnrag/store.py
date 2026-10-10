"""Snapshot knowledge, current authorization, and buffered output finality.

This is a trusted in-process API. Generation runs outside database transactions;
only the synchronous output handoff is serialized with revocation commits.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from vaultkv.ledger import Ledger
from vaultkv.models import CacheProfile, ConflictError, Identity, Scope, content_hash, require_name
from vaultkv.txnrag.models import (
    Artifact,
    ArtifactKind,
    EvidenceRef,
    ReadMode,
    Resource,
    TxnRejected,
    TxnToken,
)
from vaultkv.txnrag.schema import SCHEMA
from vaultkv.txnrag.telemetry import record_event

T = TypeVar("T")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ref(value: str) -> EvidenceRef:
    data = json.loads(value)
    data["resource"] = Resource(**data["resource"])
    return EvidenceRef(**data)


def _descriptor(artifact: Artifact) -> str:
    data = asdict(artifact)
    del data["payload"]
    return _json(data)


class TxnRAG:
    """Local coordinator with retained immutable snapshots and explicit ACLs."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._ledger = Ledger(path)
        with self._ledger.transaction() as db:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    db.execute(statement)

    def __enter__(self) -> TxnRAG:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._ledger.close()

    @property
    def knowledge_epoch(self) -> int:
        with self._ledger.transaction() as db:
            return db.execute("SELECT knowledge_epoch FROM txn_meta").fetchone()[0]

    def resource_epoch(self, resource: Resource) -> int:
        with self._ledger.transaction() as db:
            return self._resource(db, resource)["auth_epoch"]

    @staticmethod
    def _resource(db: sqlite3.Connection, resource: Resource) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM txn_resources WHERE tenant_ns = ? AND doc_id = ?",
            (resource.tenant_ns, resource.doc_id),
        ).fetchone()
        if row is None:
            raise KeyError(resource)
        return row

    def publish(
        self,
        updates: Mapping[Resource, Mapping[str, str]],
        *,
        expected_epoch: int,
        deletes: Iterable[Resource] = (),
    ) -> int:
        """Atomically publish a corpus delta; updates never overwrite current ACLs.

        Deletion removes a resource from new snapshots. Existing snapshots retain
        it subject to CURRENT authorization. Revoke separately for immediate denial.
        New resources start workspace-scoped with an empty access list (default deny).
        """
        removed = set(deletes)
        if removed.intersection(updates):
            raise ValueError("A publication cannot update and delete the same resource")
        prepared = {}
        for resource, chunks in updates.items():
            rows = []
            for chunk_id, text in chunks.items():
                require_name(chunk_id)
                rows.append((chunk_id, content_hash(text), text))
            prepared[resource] = rows
        with self._ledger.transaction() as db:
            current = db.execute("SELECT knowledge_epoch FROM txn_meta").fetchone()[0]
            if current != expected_epoch:
                raise ConflictError("Knowledge epoch changed; refresh before publishing")
            if not prepared and not removed:
                return current
            epoch = current + 1
            db.execute("INSERT INTO txn_snapshots VALUES (?)", (epoch,))
            db.execute(
                "INSERT INTO txn_snapshot_docs SELECT ?, tenant_ns, doc_id, doc_version "
                "FROM txn_snapshot_docs WHERE epoch = ?",
                (epoch, current),
            )
            for resource in removed | prepared.keys():
                db.execute(
                    "DELETE FROM txn_snapshot_docs "
                    "WHERE epoch = ? AND tenant_ns = ? AND doc_id = ?",
                    (epoch, resource.tenant_ns, resource.doc_id),
                )
            for resource, chunks in prepared.items():
                db.execute(
                    "INSERT INTO txn_resources VALUES (?, ?, 1, 1, ?) "
                    "ON CONFLICT(tenant_ns, doc_id) DO UPDATE "
                    "SET latest_version = latest_version + 1",
                    (resource.tenant_ns, resource.doc_id, Scope.TENANT),
                )
                version = self._resource(db, resource)["latest_version"]
                db.execute(
                    "INSERT INTO txn_snapshot_docs VALUES (?, ?, ?, ?)",
                    (epoch, resource.tenant_ns, resource.doc_id, version),
                )
                db.executemany(
                    "INSERT INTO txn_chunks VALUES (?, ?, ?, ?, ?, ?)",
                    [(resource.tenant_ns, resource.doc_id, version, *chunk) for chunk in chunks],
                )
            db.execute("UPDATE txn_meta SET knowledge_epoch = ?", (epoch,))
            record_event(
                db,
                "knowledge.published",
                {
                    "from_epoch": current,
                    "to_epoch": epoch,
                    "updated": [asdict(resource) for resource in prepared],
                    "deleted": [asdict(resource) for resource in sorted(removed)],
                    "cache_effect": "new_versions_use_new_keys; pinned_snapshots_retained",
                },
            )
            return epoch

    def set_permissions(
        self,
        resource: Resource,
        *,
        expected_epoch: int,
        principals: Iterable[Identity] = (),
        groups: Iterable[str] = (),
        scope: Scope = Scope.TENANT,
    ) -> int:
        scope = Scope(scope)
        members = Ledger._validate_policy(resource.tenant_ns, scope, principals)
        group_ids = frozenset(groups)
        for group_id in group_ids:
            require_name(group_id)
        if scope == Scope.PUBLIC and group_ids:
            raise ValueError("Public documents do not use an allowed-group list")
        with self._ledger.transaction() as db:
            row = self._resource(db, resource)
            if row["auth_epoch"] != expected_epoch:
                raise ConflictError("Resource authorization epoch changed")
            known_groups = {
                item[0]
                for item in db.execute(
                    "SELECT group_id FROM txn_groups WHERE tenant_ns = ?",
                    (resource.tenant_ns,),
                )
            }
            unknown = group_ids - known_groups
            if unknown:
                raise ValueError(f"Unknown groups: {', '.join(sorted(unknown))}")
            old = {
                item[0]
                for item in db.execute(
                    "SELECT principal_id FROM txn_acl WHERE tenant_ns = ? AND doc_id = ?",
                    (resource.tenant_ns, resource.doc_id),
                )
            }
            old_groups = {
                item[0]
                for item in db.execute(
                    "SELECT group_id FROM txn_acl_groups WHERE tenant_ns = ? AND doc_id = ?",
                    (resource.tenant_ns, resource.doc_id),
                )
            }
            if (
                old == {member.principal_id for member in members}
                and old_groups == group_ids
                and row["scope"] == scope
            ):
                return expected_epoch
            for member in members:
                self._principal(db, member)
            db.execute(
                "DELETE FROM txn_acl WHERE tenant_ns = ? AND doc_id = ?",
                (resource.tenant_ns, resource.doc_id),
            )
            db.execute(
                "DELETE FROM txn_acl_groups WHERE tenant_ns = ? AND doc_id = ?",
                (resource.tenant_ns, resource.doc_id),
            )
            db.executemany(
                "INSERT INTO txn_acl VALUES (?, ?, ?)",
                [(resource.tenant_ns, resource.doc_id, member.principal_id) for member in members],
            )
            db.executemany(
                "INSERT INTO txn_acl_groups VALUES (?, ?, ?)",
                [(resource.tenant_ns, resource.doc_id, group_id) for group_id in group_ids],
            )
            db.execute(
                "UPDATE txn_resources SET scope = ?, auth_epoch = auth_epoch + 1 "
                "WHERE tenant_ns = ? AND doc_id = ?",
                (scope, resource.tenant_ns, resource.doc_id),
            )
            record_event(
                db,
                "authorization.resource_changed",
                {
                    "resource": asdict(resource),
                    "from_epoch": expected_epoch,
                    "to_epoch": expected_epoch + 1,
                    "scope": scope,
                    "principal_count": len(members),
                    "group_count": len(group_ids),
                    "cache_effect": "old_authorization_epochs_rejected",
                },
            )
            return expected_epoch + 1

    def set_group(
        self,
        tenant_ns: str,
        group_id: str,
        *,
        members: Iterable[Identity] = (),
        enabled: bool = True,
        expected_epoch: int = 0,
    ) -> int:
        """Create or update a reusable group and invalidate affected document reads."""
        require_name(tenant_ns)
        require_name(group_id)
        identities = frozenset(members)
        if any(member.tenant_ns != tenant_ns for member in identities):
            raise ValueError("Group members must belong to the same workspace")
        with self._ledger.transaction() as db:
            row = db.execute(
                "SELECT * FROM txn_groups WHERE tenant_ns = ? AND group_id = ?",
                (tenant_ns, group_id),
            ).fetchone()
            current = row["auth_epoch"] if row else 0
            if current != expected_epoch:
                raise ConflictError("Group access version changed")
            old_members = {
                item[0]
                for item in db.execute(
                    "SELECT principal_id FROM txn_group_members "
                    "WHERE tenant_ns = ? AND group_id = ?",
                    (tenant_ns, group_id),
                )
            }
            member_ids = {member.principal_id for member in identities}
            if row and old_members == member_ids and bool(row["enabled"]) == bool(enabled):
                return current
            for member in identities:
                self._principal(db, member)
            if row:
                db.execute(
                    "UPDATE txn_groups SET auth_epoch = auth_epoch + 1, enabled = ? "
                    "WHERE tenant_ns = ? AND group_id = ?",
                    (int(enabled), tenant_ns, group_id),
                )
            else:
                db.execute(
                    "INSERT INTO txn_groups VALUES (?, ?, 1, ?)",
                    (tenant_ns, group_id, int(enabled)),
                )
            db.execute(
                "DELETE FROM txn_group_members WHERE tenant_ns = ? AND group_id = ?",
                (tenant_ns, group_id),
            )
            db.executemany(
                "INSERT INTO txn_group_members VALUES (?, ?, ?)",
                [(tenant_ns, group_id, principal_id) for principal_id in member_ids],
            )
            affected = [
                item[0]
                for item in db.execute(
                    "SELECT doc_id FROM txn_acl_groups WHERE tenant_ns = ? AND group_id = ?",
                    (tenant_ns, group_id),
                )
            ]
            db.executemany(
                "UPDATE txn_resources SET auth_epoch = auth_epoch + 1 "
                "WHERE tenant_ns = ? AND doc_id = ?",
                [(tenant_ns, doc_id) for doc_id in affected],
            )
            record_event(
                db,
                "authorization.group_changed",
                {
                    "tenant_ns": tenant_ns,
                    "group_id": group_id,
                    "from_epoch": current,
                    "to_epoch": current + 1,
                    "enabled": bool(enabled),
                    "member_count": len(member_ids),
                    "affected_documents": len(affected),
                },
            )
            return current + 1

    def access_directory(self, tenant_ns: str) -> dict[str, object]:
        """List configured users and reusable groups for a workspace."""
        require_name(tenant_ns)
        with self._ledger.transaction() as db:
            names = {
                row[0]
                for row in db.execute(
                    "SELECT principal_id FROM txn_principals WHERE tenant_ns = ? "
                    "UNION SELECT principal_id FROM txn_acl WHERE tenant_ns = ? "
                    "UNION SELECT principal_id FROM txn_group_members WHERE tenant_ns = ?",
                    (tenant_ns, tenant_ns, tenant_ns),
                )
            }
            for name in names:
                self._principal(db, Identity(tenant_ns, name))
            users = [
                {
                    "user": row["principal_id"],
                    "enabled": bool(row["enabled"]),
                    "access_version": row["auth_epoch"],
                }
                for row in db.execute(
                    "SELECT * FROM txn_principals WHERE tenant_ns = ? ORDER BY principal_id",
                    (tenant_ns,),
                )
            ]
            groups = []
            for row in db.execute(
                "SELECT * FROM txn_groups WHERE tenant_ns = ? ORDER BY group_id",
                (tenant_ns,),
            ):
                groups.append(
                    {
                        "group": row["group_id"],
                        "enabled": bool(row["enabled"]),
                        "access_version": row["auth_epoch"],
                        "members": [
                            item[0]
                            for item in db.execute(
                                "SELECT principal_id FROM txn_group_members "
                                "WHERE tenant_ns = ? AND group_id = ? ORDER BY principal_id",
                                (tenant_ns, row["group_id"]),
                            )
                        ],
                    }
                )
            return {"workspace": tenant_ns, "users": users, "groups": groups}

    @staticmethod
    def _principal(db: sqlite3.Connection, principal: Identity) -> sqlite3.Row:
        db.execute(
            "INSERT OR IGNORE INTO txn_principals VALUES (?, ?, 1, 1)",
            (principal.tenant_ns, principal.principal_id),
        )
        return db.execute(
            "SELECT * FROM txn_principals WHERE tenant_ns = ? AND principal_id = ?",
            (principal.tenant_ns, principal.principal_id),
        ).fetchone()

    def set_principal_enabled(
        self, principal: Identity, *, enabled: bool, expected_epoch: int
    ) -> int:
        with self._ledger.transaction() as db:
            row = self._principal(db, principal)
            if row["auth_epoch"] != expected_epoch:
                raise ConflictError("User access version changed")
            if bool(row["enabled"]) == enabled:
                return expected_epoch
            db.execute(
                "UPDATE txn_principals SET enabled = ?, auth_epoch = auth_epoch + 1 "
                "WHERE tenant_ns = ? AND principal_id = ?",
                (int(enabled), principal.tenant_ns, principal.principal_id),
            )
            record_event(
                db,
                "authorization.principal_changed",
                {
                    "principal": asdict(principal),
                    "from_epoch": expected_epoch,
                    "to_epoch": expected_epoch + 1,
                    "enabled": enabled,
                },
            )
            return expected_epoch + 1

    def bump_policy_epoch(self, *, expected_epoch: int) -> int:
        """Invalidate active requests after an external policy-engine change."""
        with self._ledger.transaction() as db:
            current = db.execute("SELECT policy_epoch FROM txn_meta").fetchone()[0]
            if current != expected_epoch:
                raise ConflictError("Policy epoch changed")
            db.execute("UPDATE txn_meta SET policy_epoch = policy_epoch + 1")
            record_event(
                db, "authorization.policy_changed", {"from_epoch": current, "to_epoch": current + 1}
            )
            return current + 1

    def begin(
        self, principal: Identity, profile: CacheProfile, *, mode: ReadMode = ReadMode.SNAPSHOT
    ) -> TxnToken:
        with self._ledger.transaction() as db:
            person = self._principal(db, principal)
            if not person["enabled"]:
                raise TxnRejected("principal_disabled")
            meta = db.execute("SELECT * FROM txn_meta").fetchone()
            token = TxnToken(
                uuid4().hex,
                meta["knowledge_epoch"],
                principal,
                person["auth_epoch"],
                meta["policy_epoch"],
                profile,
                ReadMode(mode),
            )
            db.execute(
                "INSERT INTO txn_requests VALUES (?, ?, 'active')",
                (token.request_id, _json(asdict(token))),
            )
            return token

    @staticmethod
    def _check_token(db: sqlite3.Connection, token: TxnToken) -> None:
        row = db.execute(
            "SELECT * FROM txn_requests WHERE request_id = ?", (token.request_id,)
        ).fetchone()
        if row is None or row["token_json"] != _json(asdict(token)):
            raise TxnRejected("unknown_or_modified_token")
        if row["state"] != "active":
            raise TxnRejected("request_closed")
        current = db.execute("SELECT knowledge_epoch FROM txn_meta").fetchone()[0]
        if token.mode == ReadMode.FRESHEST and current != token.knowledge_epoch:
            raise TxnRejected("knowledge_epoch_changed")

    @staticmethod
    def _check_identity(db: sqlite3.Connection, token: TxnToken) -> None:
        row = TxnRAG._principal(db, token.principal)
        if not row["enabled"] or row["auth_epoch"] != token.principal_auth_epoch:
            raise TxnRejected("principal_authorization_changed")
        policy = db.execute("SELECT policy_epoch FROM txn_meta").fetchone()[0]
        if policy != token.policy_epoch:
            raise TxnRejected("policy_epoch_changed")

    @staticmethod
    def _authorize(db: sqlite3.Connection, token: TxnToken, resource: Resource) -> sqlite3.Row:
        row = TxnRAG._resource(db, resource)
        if row["scope"] == Scope.PUBLIC:
            return row
        if token.principal.tenant_ns != resource.tenant_ns:
            raise TxnRejected("permission_denied")
        allowed = db.execute(
            "SELECT 1 FROM txn_acl WHERE tenant_ns = ? AND doc_id = ? AND principal_id = ?",
            (resource.tenant_ns, resource.doc_id, token.principal.principal_id),
        ).fetchone()
        group_allowed = db.execute(
            "SELECT 1 FROM txn_acl_groups a "
            "JOIN txn_groups g USING(tenant_ns, group_id) "
            "JOIN txn_group_members m USING(tenant_ns, group_id) "
            "WHERE a.tenant_ns = ? AND a.doc_id = ? AND m.principal_id = ? "
            "AND g.enabled = 1",
            (resource.tenant_ns, resource.doc_id, token.principal.principal_id),
        ).fetchone()
        if allowed is None and group_allowed is None:
            raise TxnRejected("permission_denied")
        return row

    @staticmethod
    def _dependencies(db: sqlite3.Connection, token: TxnToken) -> tuple[EvidenceRef, ...]:
        return tuple(
            _ref(row[0])
            for row in db.execute(
                "SELECT ref_json FROM txn_dependencies WHERE request_id = ? ORDER BY ref_json",
                (token.request_id,),
            )
        )

    @staticmethod
    def _validate_ref(db: sqlite3.Connection, token: TxnToken, ref: EvidenceRef) -> None:
        if ref.knowledge_epoch != token.knowledge_epoch:
            raise TxnRejected("mixed_knowledge_snapshots")
        row = db.execute(
            "SELECT c.content_hash FROM txn_snapshot_docs s JOIN txn_chunks c "
            "ON s.tenant_ns = c.tenant_ns AND s.doc_id = c.doc_id "
            "AND s.doc_version = c.doc_version "
            "WHERE s.epoch = ? AND s.tenant_ns = ? AND s.doc_id = ? "
            "AND s.doc_version = ? AND c.chunk_id = ?",
            (
                token.knowledge_epoch,
                ref.resource.tenant_ns,
                ref.resource.doc_id,
                ref.doc_version,
                ref.chunk_id,
            ),
        ).fetchone()
        if row is None or row[0] != ref.content_hash:
            raise TxnRejected("incompatible_evidence")

    def _guard(self, db: sqlite3.Connection, token: TxnToken) -> None:
        self._check_token(db, token)
        self._check_identity(db, token)
        for ref in self._dependencies(db, token):
            self._validate_ref(db, token, ref)
            current = self._authorize(db, token, ref.resource)
            if current["auth_epoch"] != ref.resource_auth_epoch:
                raise TxnRejected("resource_authorization_changed")

    @staticmethod
    def _register(
        db: sqlite3.Connection,
        token: TxnToken,
        kind: ArtifactKind,
        dependencies: Iterable[EvidenceRef],
        payload: bytes,
    ) -> Artifact:
        if not isinstance(payload, bytes):
            raise TypeError("Artifact payloads must be bytes")
        refs = tuple(sorted(set(dependencies), key=lambda ref: _json(asdict(ref))))
        artifact = Artifact(
            uuid4().hex,
            token.request_id,
            kind,
            token.knowledge_epoch,
            token.profile.fingerprint,
            refs,
            payload,
        )
        db.execute(
            "INSERT INTO txn_artifacts VALUES (?, ?, ?, ?)",
            (artifact.artifact_id, token.request_id, _descriptor(artifact), payload),
        )
        return artifact

    def _check_artifact(self, db: sqlite3.Connection, token: TxnToken, artifact: Artifact) -> None:
        if (
            artifact.request_id != token.request_id
            or artifact.knowledge_epoch != token.knowledge_epoch
        ):
            raise TxnRejected("cross_request_artifact")
        if artifact.profile_fingerprint != token.profile.fingerprint:
            raise TxnRejected("provenance_mismatch")
        row = db.execute(
            "SELECT * FROM txn_artifacts WHERE artifact_id = ?", (artifact.artifact_id,)
        ).fetchone()
        if (
            row is None
            or row["descriptor_json"] != _descriptor(artifact)
            or row["payload"] != artifact.payload
        ):
            raise TxnRejected("unknown_or_modified_artifact")
        for ref in artifact.dependencies:
            self._validate_ref(db, token, ref)

    def retrieve(self, token: TxnToken, resources: Iterable[Resource]) -> Artifact:
        """Fetch explicitly selected resources at K, with current ACLs before text load.

        This CPU reference retriever is keyed, not a vector or BM25 search engine.
        All requested resources must be accessible; it never silently drops a denial.
        """
        with self._ledger.transaction() as db:
            self._guard(db, token)
            refs, evidence = [], []
            for resource in sorted(set(resources)):
                current = self._authorize(db, token, resource)
                version = db.execute(
                    "SELECT doc_version FROM txn_snapshot_docs "
                    "WHERE epoch = ? AND tenant_ns = ? AND doc_id = ?",
                    (token.knowledge_epoch, resource.tenant_ns, resource.doc_id),
                ).fetchone()
                if version is None:
                    raise TxnRejected("resource_not_in_snapshot")
                for chunk in db.execute(
                    "SELECT * FROM txn_chunks WHERE tenant_ns = ? AND doc_id = ? "
                    "AND doc_version = ? ORDER BY chunk_id",
                    (resource.tenant_ns, resource.doc_id, version[0]),
                ):
                    ref = EvidenceRef(
                        resource,
                        version[0],
                        chunk["chunk_id"],
                        chunk["content_hash"],
                        token.knowledge_epoch,
                        current["auth_epoch"],
                    )
                    refs.append(ref)
                    evidence.append({"source": asdict(ref), "text": chunk["text"]})
                    db.execute(
                        "INSERT OR IGNORE INTO txn_dependencies VALUES (?, ?)",
                        (token.request_id, _json(asdict(ref))),
                    )
            return self._register(db, token, ArtifactKind.EVIDENCE, refs, _json(evidence).encode())

    def derive(
        self, token: TxnToken, kind: ArtifactKind, parents: Iterable[Artifact], payload: bytes
    ) -> Artifact:
        """Record trusted processing; propagate all parent evidence dependencies.

        This records private intermediate results, not authorization to emit them.
        It intentionally permits a worker to finish buffering after revocation.
        """
        kind = ArtifactKind(kind)
        if kind in (ArtifactKind.EVIDENCE, ArtifactKind.KV):
            raise ValueError("Evidence and KV artifacts require their dedicated APIs")
        parents = tuple(parents)
        if not parents:
            raise ValueError("Derived artifacts require at least one parent")
        with self._ledger.transaction() as db:
            self._check_token(db, token)
            dependencies = set()
            for parent in parents:
                self._check_artifact(db, token, parent)
                dependencies.update(parent.dependencies)
            return self._register(db, token, kind, dependencies, payload)

    def release(self, token: TxnToken, answer: Artifact, sink: Callable[[bytes], T]) -> T:
        """Check current authorization and hand off buffered output under one lock.

        The sink must synchronously complete the protected handoff. No streaming,
        async scheduling, retained buffer, or reentrant calls are supported.
        Sink errors permanently close the request, avoiding automatic replay.
        """
        if inspect.iscoroutinefunction(sink):
            raise TypeError("Output sinks must be synchronous")
        failure: BaseException | None = None
        with self._ledger.transaction() as db:
            self._guard(db, token)
            self._check_artifact(db, token, answer)
            if answer.kind != ArtifactKind.ANSWER:
                raise TxnRejected("not_an_answer")
            if set(answer.dependencies) != set(self._dependencies(db, token)):
                raise TxnRejected("incomplete_answer_dependencies")
            try:
                result = sink(answer.payload)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("Output sinks must not return awaitables")
            except BaseException as exc:
                failure = exc
            db.execute(
                "UPDATE txn_requests SET state = ? WHERE request_id = ?",
                ("failed" if failure else "emitted", token.request_id),
            )
        if failure is not None:
            raise failure
        return result

    def cancel(self, token: TxnToken) -> None:
        with self._ledger.transaction() as db:
            # Cancellation must also work after freshest-mode invalidation.
            row = db.execute(
                "SELECT token_json FROM txn_requests WHERE request_id = ?", (token.request_id,)
            ).fetchone()
            if row is None or row[0] != _json(asdict(token)):
                raise TxnRejected("unknown_or_modified_token")
            db.execute(
                "UPDATE txn_requests SET state = 'cancelled' "
                "WHERE request_id = ? AND state = 'active'",
                (token.request_id,),
            )
