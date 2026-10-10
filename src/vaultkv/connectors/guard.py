"""Durable permits shared by the gateway, vLLM scheduler, and GPU worker.

Only trusted producers can issue permits. Possessing a permit never replaces
current authorization. A guard holds the SQLite write lock through each transfer.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict

from vaultkv.models import CacheProfile, Identity, digest
from vaultkv.txnrag import Artifact, ArtifactKind, ReadMode, TxnRAG, TxnRejected, TxnToken
from vaultkv.txnrag.telemetry import record_event

PERMIT = "lmcache.vaultkv_permit"
NAMESPACE = "lmcache.tag.vaultkv"


def token_from_json(raw: str) -> TxnToken:
    data = json.loads(raw)
    data["principal"] = Identity(**data["principal"])
    data["profile"] = CacheProfile(**data["profile"])
    data["mode"] = ReadMode(data["mode"])
    return TxnToken(**data)


class TransferGuard:
    def __init__(self, store: TxnRAG) -> None:
        self.store = store

    def issue(self, token: TxnToken, context: Artifact, token_ids: list[int]) -> dict[str, str]:
        if not token_ids or any(type(value) is not int or value < 0 for value in token_ids):
            raise ValueError("Expected nonempty integer prompt token IDs")
        with self.store._ledger.transaction() as db:
            self.store._guard(db, token)
            self.store._check_artifact(db, token, context)
            if context.kind != ArtifactKind.CONTEXT:
                raise TxnRejected("not_a_context")
            if set(context.dependencies) != set(self.store._dependencies(db, token)):
                raise TxnRejected("incomplete_context_dependencies")
            sources = []
            for ref in context.dependencies:
                item = asdict(ref)
                del item["knowledge_epoch"]
                sources.append(item)
            namespace = digest(
                {
                    "protocol": "vaultkv-lmcache-prefix-v1",
                    "principal": asdict(token.principal),
                    "principal_epoch": token.principal_auth_epoch,
                    "policy_epoch": token.policy_epoch,
                    "profile": token.profile.fingerprint,
                    "sources": sources,
                    "context": digest(context.payload.decode()),
                    "tokens": token_ids,
                }
            )
            permit = secrets.token_urlsafe(32)
            db.execute(
                "INSERT INTO txn_connector_permits VALUES (?, ?, ?, ?, ?)",
                (permit, token.request_id, context.artifact_id, namespace, json.dumps(token_ids)),
            )
            record_event(db, "connector.permit_issued", {"namespace": namespace}, token.request_id)
            return {PERMIT: permit, NAMESPACE: namespace}

    def validate(self, db, configs: dict, token_ids: Sequence[int], *, prefix: bool = False):
        if not isinstance(configs, dict) or set(configs) != {PERMIT, NAMESPACE}:
            raise TxnRejected("invalid_connector_parameters")
        if any(not isinstance(value, str) for value in configs.values()):
            raise TxnRejected("invalid_connector_parameters")
        row = db.execute(
            "SELECT p.*, r.token_json FROM txn_connector_permits p "
            "JOIN txn_requests r USING(request_id) WHERE permit = ?",
            (configs[PERMIT],),
        ).fetchone()
        if row is None or row["namespace"] != configs[NAMESPACE]:
            raise TxnRejected("unknown_connector_permit")
        token = token_from_json(row["token_json"])
        self.store._guard(db, token)
        expected = json.loads(row["token_ids_json"])
        supplied = list(token_ids)
        if not supplied or any(type(value) is not int or value < 0 for value in supplied):
            raise TxnRejected("connector_prompt_mismatch")
        if (not prefix and supplied != expected) or (
            prefix and (len(supplied) > len(expected) or supplied != expected[: len(supplied)])
        ):
            raise TxnRejected("connector_prompt_mismatch")
        context = db.execute(
            "SELECT descriptor_json FROM txn_artifacts WHERE artifact_id = ?",
            (row["context_id"],),
        ).fetchone()
        if context is None:
            raise TxnRejected("unknown_context")
        refs = json.loads(context[0])["dependencies"]
        current_refs = [asdict(ref) for ref in self.store._dependencies(db, token)]
        if sorted(map(digest, refs)) != sorted(map(digest, current_refs)):
            raise TxnRejected("connector_dependencies_changed")
        return token

    @contextmanager
    def checked(self, requests: Sequence[tuple[dict, Sequence[int]]], *, prefix=False) -> Iterator:
        with self.store._ledger.transaction() as db:
            tokens = [self.validate(db, configs, ids, prefix=prefix) for configs, ids in requests]
            yield db, tokens

    def close(self, configs: dict) -> None:
        with self.store._ledger.transaction() as db:
            db.execute("DELETE FROM txn_connector_permits WHERE permit = ?", (configs[PERMIT],))
