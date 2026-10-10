"""Seeded CPU correctness workload with state independent of ledger reads."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from vaultkv import CacheProfile, Identity, Ledger, Provenance, Query, Rejected, Scope, Status
from vaultkv.cache import VaultCache
from vaultkv.models import Ticket, content_hash


@dataclass
class _Truth:
    tenant_ns: str
    doc_id: str
    version: int = 1
    epoch: int = 1
    text: str = "initial"
    deleted: bool = False
    members: set[str] = field(default_factory=lambda: {"alice", "bob"})

    def query(self, principal: str) -> Query:
        return Query(
            self.tenant_ns,
            self.doc_id,
            "chunk",
            self.version,
            self.epoch,
            content_hash(self.text),
            Identity(self.tenant_ns, principal),
        )

    def expected(self, query: Query) -> Status | None:
        if self.deleted or query.requester.principal_id not in self.members:
            return Status.DENIED
        if query.doc_version != self.version or query.policy_epoch != self.epoch:
            return Status.STALE
        if query.content_hash != content_hash(self.text):
            return Status.STALE
        return None  # Authorized/current; either hit or miss is correct.

    def payload(self) -> bytes:
        return f"{self.tenant_ns}/{self.doc_id}/v{self.version}/{self.text}".encode()


def run_workload(*, steps: int = 1000, seed: int = 42) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = random.Random(seed)
    truth = {
        (f"tenant-{tenant}", f"doc-{doc}"): _Truth(f"tenant-{tenant}", f"doc-{doc}")
        for tenant in range(3)
        for doc in range(4)
    }
    profile = CacheProfile("synthetic-model@1", "synthetic-tokenizer@1", "bytes-v1", "canonical")
    provenance = Provenance(profile.context_hash)
    history: list[Query] = []
    pending: list[Ticket] = []
    counts: Counter[str] = Counter()

    def check_consumption(cache: VaultCache, ticket: Ticket) -> None:
        state = truth[(ticket.query.owner_ns, ticket.query.doc_id)]
        expected = state.expected(ticket.query)

        def consume(payload: bytes) -> None:
            # The oracle never reads ledger metadata or calls admission to decide safety.
            if expected is not None or payload != state.payload():
                raise AssertionError("Unsafe serve detected by independent workload oracle")
            counts["safe_consumptions"] += 1

        try:
            cache.consume(ticket, consume)
        except Rejected as exc:
            if expected is not None:
                if exc.decision.status != expected:
                    raise AssertionError(f"Expected {expected}, got {exc.decision}") from exc
            elif exc.decision.status != Status.MISS:
                raise AssertionError(f"Unexpected consumption rejection: {exc.decision}") from exc
            counts["queued_rejections"] += 1

    with Ledger() as ledger:
        cache = VaultCache(ledger)
        for state in truth.values():
            ledger.put_document(
                state.tenant_ns,
                state.doc_id,
                {"chunk": state.text},
                expected_version=0,
                principals=[Identity(state.tenant_ns, name) for name in sorted(state.members)],
            )
        for step in range(steps):
            state = rng.choice(list(truth.values()))
            event = rng.choices(["query", "edit", "acl", "delete"], weights=[6, 2, 2, 1])[0]
            if event == "query":
                query = (
                    rng.choice(history)
                    if history and rng.random() < 0.3
                    else state.query(rng.choice(["alice", "bob", "eve"]))
                )
                history.append(query)
                state = truth[(query.owner_ns, query.doc_id)]
                expected = state.expected(query)
                decision = cache.admit(query, profile)
                if expected is None:
                    if decision.status not in (Status.HIT, Status.MISS):
                        raise AssertionError(f"Unexpected rejection: {decision}")
                    ticket = decision.ticket or cache.publish(
                        query, profile, provenance, state.payload()
                    )
                    check_consumption(cache, ticket)
                    pending.append(ticket)
                elif decision.status != expected:
                    raise AssertionError(f"Expected {expected}, got {decision}")
                counts[decision.status.value] += 1
            elif event == "edit":
                was_deleted = state.deleted
                new_text = f"revision-{step}"
                ledger.put_document(
                    state.tenant_ns,
                    state.doc_id,
                    {"chunk": new_text},
                    expected_version=state.version,
                    expected_epoch=state.epoch,
                    principals=[Identity(state.tenant_ns, name) for name in sorted(state.members)],
                )
                state.version += 1
                state.epoch += int(was_deleted)
                state.deleted = False
                state.text = new_text
                counts["edits"] += 1
            elif event == "acl" and not state.deleted:
                members = state.members ^ {rng.choice(["alice", "bob"])}
                ledger.set_permissions(
                    state.tenant_ns,
                    state.doc_id,
                    expected_epoch=state.epoch,
                    acl_scope=Scope.TENANT,
                    principals=[Identity(state.tenant_ns, name) for name in sorted(members)],
                )
                state.members = members
                state.epoch += 1
                counts["acl_changes"] += 1
            elif event == "delete" and not state.deleted:
                ledger.delete_document(
                    state.tenant_ns, state.doc_id, expected_version=state.version
                )
                state.version += 1
                state.epoch += 1
                state.deleted = True
                state.members.clear()
                counts["deletions"] += 1
            if pending:
                check_consumption(cache, rng.choice(pending))
    return {
        "mode": "synthetic-cpu-correctness",
        "seed": seed,
        "steps": steps,
        "unsafe_serves": 0,
        "counts": dict(sorted(counts.items())),
    }
