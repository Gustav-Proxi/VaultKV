"""Local opaque-byte cache with a guarded consumption boundary."""

from collections.abc import Callable
from typing import TypeVar

from vaultkv.admission import cache_key, evaluate
from vaultkv.ledger import Ledger
from vaultkv.models import CacheProfile, Decision, Provenance, Query, Rejected, Status, Ticket

T = TypeVar("T")


class VaultCache:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def admit(self, query: Query, profile: CacheProfile) -> Decision:
        with self.ledger.transaction() as db:
            return evaluate(db, query, profile)

    def publish(
        self, query: Query, profile: CacheProfile, provenance: Provenance, payload: bytes
    ) -> Ticket:
        """Publish only from a trusted producer; supplied provenance is an assertion.

        These checks cannot prove how an opaque tensor was computed. A future GPU
        adapter must own canonical prefill and never accept client-produced KV.
        """
        if provenance.position != 0 or provenance.context_hash != profile.context_hash:
            raise ValueError("Cache publication requires canonical position-zero provenance")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("Cache payload must be nonempty bytes")
        with self.ledger.transaction() as db:
            decision = evaluate(db, query, profile)
            if decision.status not in (Status.HIT, Status.MISS):
                raise Rejected(decision)
            key = cache_key(db, query, profile)
            # First canonical writer wins; a later publisher cannot replace live bytes.
            db.execute(
                "INSERT OR IGNORE INTO cache_entries VALUES (?, ?, ?, ?)",
                (key, query.owner_ns, query.doc_id, payload),
            )
            return Ticket(query, profile, key)

    def consume(self, ticket: Ticket, consumer: Callable[[bytes], T]) -> T:
        """Revalidate and synchronously consume under the mutation lock.

        The callback must finish all protected use/output before returning and must
        not retain bytes, dispatch async work, or call back into this ledger. Edits
        and revocations wait for active consumption; they take effect on commit.
        """
        with self.ledger.transaction() as db:
            decision = evaluate(db, ticket.query, ticket.profile)
            if decision.status != Status.HIT:
                raise Rejected(decision)
            if decision.ticket != ticket:
                raise Rejected(Decision(Status.DENIED, "ticket_mismatch"))
            payload = db.execute(
                "SELECT payload FROM cache_entries WHERE cache_key = ?", (ticket.cache_key,)
            ).fetchone()[0]
            return consumer(payload)
