"""Immutable metadata passed between trusted ingestion and serving components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(parts: object) -> str:
    """Structured encoding avoids ambiguous concatenated cache identities."""
    return content_hash(json.dumps(parts, sort_keys=True, separators=(",", ":")))


def require_name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Identifiers must be nonempty strings")


class Scope(StrEnum):
    PUBLIC = "public"
    TENANT = "tenant-shared"
    PRIVATE = "private"


class Status(StrEnum):
    HIT = "hit"
    MISS = "miss"
    DENIED = "denied"
    STALE = "stale"


@dataclass(frozen=True)
class Identity:
    tenant_ns: str
    principal_id: str

    def __post_init__(self) -> None:
        require_name(self.tenant_ns)
        require_name(self.principal_id)


@dataclass(frozen=True)
class CacheProfile:
    """Exact producer compatibility, including immutable model/tokenizer revisions.

    kv_format must include tensor layout, dtype, and relevant engine configuration.
    context_hash identifies the trusted canonical prefill template/token context.
    """

    model_revision: str
    tokenizer_revision: str
    kv_format: str
    context_hash: str
    adapter_revision: str = "none"

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            require_name(value)

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Provenance:
    context_hash: str
    position: int = 0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content_hash: str


@dataclass(frozen=True)
class Query:
    owner_ns: str
    doc_id: str
    chunk_id: str
    doc_version: int
    policy_epoch: int
    content_hash: str
    requester: Identity


@dataclass(frozen=True)
class Document:
    tenant_ns: str
    doc_id: str
    doc_version: int
    policy_epoch: int
    acl_scope: Scope
    deleted: bool
    chunks: tuple[Chunk, ...]
    principals: frozenset[Identity]

    def query(self, chunk_id: str, requester: Identity) -> Query:
        chunk = next((item for item in self.chunks if item.chunk_id == chunk_id), None)
        if chunk is None:
            raise KeyError(chunk_id)
        return Query(
            self.tenant_ns,
            self.doc_id,
            chunk_id,
            self.doc_version,
            self.policy_epoch,
            chunk.content_hash,
            requester,
        )


@dataclass(frozen=True)
class Ticket:
    """Scheduling hint; always revalidated. Not an authentication credential."""

    query: Query
    profile: CacheProfile
    cache_key: str


@dataclass(frozen=True)
class Decision:
    status: Status
    reason: str
    ticket: Ticket | None = None


class ConflictError(ValueError):
    """A concurrent edit changed the expected version or policy epoch."""


class Rejected(RuntimeError):
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(f"{decision.status}: {decision.reason}")
