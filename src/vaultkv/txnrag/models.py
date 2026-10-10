"""Immutable request contracts for the local transactional RAG prototype."""

from dataclasses import dataclass
from enum import StrEnum

from vaultkv.models import CacheProfile, Identity, require_name


class ReadMode(StrEnum):
    SNAPSHOT = "snapshot"
    FRESHEST = "freshest"


class ArtifactKind(StrEnum):
    EVIDENCE = "evidence"
    RERANKED = "reranked"
    CONTEXT = "context"
    KV = "kv"
    ANSWER = "answer"


@dataclass(frozen=True, order=True)
class Resource:
    tenant_ns: str
    doc_id: str

    def __post_init__(self) -> None:
        require_name(self.tenant_ns)
        require_name(self.doc_id)


@dataclass(frozen=True)
class TxnToken:
    request_id: str
    knowledge_epoch: int
    principal: Identity
    principal_auth_epoch: int
    policy_epoch: int
    profile: CacheProfile
    mode: ReadMode


@dataclass(frozen=True)
class EvidenceRef:
    resource: Resource
    doc_version: int
    chunk_id: str
    content_hash: str
    knowledge_epoch: int
    resource_auth_epoch: int


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    request_id: str
    kind: ArtifactKind
    knowledge_epoch: int
    profile_fingerprint: str
    dependencies: tuple[EvidenceRef, ...]
    payload: bytes


class TxnRejected(RuntimeError):
    """A request must not release output; callers may start a new transaction."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
