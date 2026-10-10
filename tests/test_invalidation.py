from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vaultkv import ConflictError, Ledger, Provenance, Rejected, Scope, Status, VaultCache


def test_edit_invalidates_queued_hit(ledger, cache, document, alice, profile):
    query = document.query("c1", alice)
    ticket = cache.publish(query, profile, Provenance(profile.context_hash), b"old")
    updated = ledger.put_document(
        "tenant-a",
        "doc",
        {"c1": "changed"},
        expected_version=1,
        expected_epoch=1,
        principals=[alice],
    )
    assert updated.doc_version == 2
    with pytest.raises(Rejected, match="version_mismatch"):
        cache.consume(ticket, lambda _: pytest.fail("Loaded stale KV"))
    with pytest.raises(Rejected):
        cache.publish(query, profile, Provenance(profile.context_hash), b"late-old-computation")
    assert cache.admit(updated.query("c1", alice), profile).status == Status.MISS


def test_revoke_blocks_queued_hit_and_refreshes_remaining_member(
    ledger, cache, document, alice, bob, profile
):
    query = document.query("c1", alice)
    ticket = cache.publish(query, profile, Provenance(profile.context_hash), b"secret")
    updated = ledger.set_permissions(
        "tenant-a", "doc", expected_epoch=1, acl_scope=Scope.TENANT, principals=[bob]
    )
    assert updated.policy_epoch == 2
    with pytest.raises(Rejected, match="permission_denied"):
        cache.consume(ticket, lambda _: pytest.fail("Consumed after revocation"))
    assert cache.admit(updated.query("c1", alice), profile).status == Status.DENIED
    assert cache.admit(document.query("c1", bob), profile).status == Status.STALE
    assert cache.admit(updated.query("c1", bob), profile).status == Status.MISS


def test_delete_recreate_does_not_resurrect_old_ticket(ledger, cache, document, alice, profile):
    ticket = cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"old"
    )
    tombstone = ledger.delete_document("tenant-a", "doc", expected_version=1)
    assert tombstone.deleted
    assert not tombstone.chunks
    assert cache.admit(ticket.query, profile).status == Status.DENIED
    recreated = ledger.put_document(
        "tenant-a",
        "doc",
        {"c1": "alpha"},
        expected_version=tombstone.doc_version,
        expected_epoch=tombstone.policy_epoch,
        principals=[alice],
    )
    assert recreated.doc_version == 3
    assert cache.admit(ticket.query, profile).status == Status.STALE
    with pytest.raises(Rejected):
        cache.consume(ticket, lambda _: pytest.fail("Old ticket resurrected"))


def test_stale_writer_cannot_undo_revocation(ledger, document, alice, bob):
    ledger.set_permissions(
        "tenant-a", "doc", expected_epoch=1, acl_scope=Scope.TENANT, principals=[bob]
    )
    with pytest.raises(ConflictError, match="epoch"):
        ledger.put_document(
            "tenant-a",
            "doc",
            {"c1": "edited by stale writer"},
            expected_version=1,
            expected_epoch=1,
            principals=[alice, bob],
        )
    assert ledger.snapshot("tenant-a", "doc").principals == frozenset([bob])


def test_regrant_does_not_restore_old_ticket(ledger, cache, document, alice, profile):
    query = document.query("c1", alice)
    ticket = cache.publish(query, profile, Provenance(profile.context_hash), b"old")
    ledger.set_permissions("tenant-a", "doc", expected_epoch=1, acl_scope=Scope.TENANT)
    with pytest.raises(Rejected, match="permission_denied"):
        cache.publish(query, profile, Provenance(profile.context_hash), b"late-producer")
    fresh = ledger.set_permissions(
        "tenant-a", "doc", expected_epoch=2, acl_scope=Scope.TENANT, principals=[alice]
    )
    with pytest.raises(Rejected, match="epoch_mismatch"):
        cache.consume(ticket, lambda _: pytest.fail("Old epoch accepted after regrant"))
    assert cache.admit(fresh.query("c1", alice), profile).status == Status.MISS


def test_conflicting_writes_are_rejected(ledger, document, alice):
    with pytest.raises(ConflictError):
        ledger.put_document("tenant-a", "doc", {}, expected_version=0)
    with pytest.raises(ConflictError):
        ledger.set_permissions("tenant-a", "doc", expected_epoch=0, acl_scope=Scope.PUBLIC)
    with pytest.raises(ConflictError):
        ledger.delete_document("tenant-a", "doc", expected_version=0)
    assert ledger.snapshot("tenant-a", "doc") == document


def test_consumer_error_releases_transaction(ledger, cache, document, alice, profile):
    ticket = cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"data"
    )

    def fail(_):
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        cache.consume(ticket, fail)
    ledger.delete_document("tenant-a", "doc", expected_version=1)
    assert cache.admit(ticket.query, profile).status == Status.DENIED


def test_reentrant_mutation_is_rejected(ledger, cache, document, alice, profile):
    ticket = cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"data"
    )
    with pytest.raises(RuntimeError, match="Reentrant"):
        cache.consume(
            ticket, lambda _: ledger.delete_document("tenant-a", "doc", expected_version=1)
        )
    assert ledger.snapshot("tenant-a", "doc").deleted is False


def test_revocation_serializes_with_active_consumer_across_connections(tmp_path, alice, profile):
    path = tmp_path / "shared.sqlite3"
    with Ledger(path) as reader, Ledger(path) as writer:
        doc = reader.put_document(
            "tenant-a", "doc", {"c1": "alpha"}, expected_version=0, principals=[alice]
        )
        cache = VaultCache(reader)
        ticket = cache.publish(
            doc.query("c1", alice), profile, Provenance(profile.context_hash), b"kv"
        )
        consuming, release, attempted, revoked = Event(), Event(), Event(), Event()

        def consume(payload):
            consuming.set()
            assert release.wait(timeout=3)
            assert not revoked.is_set()
            return len(payload)

        def revoke():
            attempted.set()
            writer.set_permissions("tenant-a", "doc", expected_epoch=1, acl_scope=Scope.TENANT)
            revoked.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            serving = pool.submit(cache.consume, ticket, consume)
            try:
                assert consuming.wait(timeout=3)
                mutation = pool.submit(revoke)
                assert attempted.wait(timeout=3)
                assert not revoked.wait(timeout=0.05)
            finally:
                release.set()
            assert serving.result(timeout=3) == 2
            mutation.result(timeout=3)
        assert revoked.is_set()
        with pytest.raises(Rejected):
            cache.consume(ticket, lambda _: pytest.fail("Consumed after committed revocation"))
