import sqlite3

import pytest

from vaultkv import Identity, Ledger, Provenance, Scope, Status, VaultCache


def test_reopen_preserves_metadata_and_bytes(tmp_path, alice, profile):
    path = tmp_path / "ledger.sqlite3"
    with Ledger(path) as ledger:
        doc = ledger.put_document(
            "tenant-a", "doc", {"c1": "alpha"}, expected_version=0, principals=[alice]
        )
        ticket = VaultCache(ledger).publish(
            doc.query("c1", alice), profile, Provenance(profile.context_hash), b"durable"
        )
    with Ledger(path) as ledger:
        assert ledger.snapshot("tenant-a", "doc") == doc
        assert VaultCache(ledger).consume(ticket, lambda payload: payload.decode()) == "durable"
        ledger.delete_document("tenant-a", "doc", expected_version=1)
    with Ledger(path) as ledger:
        assert ledger.snapshot("tenant-a", "doc").deleted
        assert VaultCache(ledger).admit(ticket.query, profile).status == Status.DENIED


def test_invalidation_removes_cached_payloads(tmp_path, alice, profile):
    path = tmp_path / "ledger.sqlite3"
    with Ledger(path) as ledger:
        doc = ledger.put_document(
            "tenant-a",
            "doc",
            {"c1": "alpha", "c2": "beta"},
            expected_version=0,
            principals=[alice],
        )
        for chunk in doc.chunks:
            VaultCache(ledger).publish(
                doc.query(chunk.chunk_id, alice), profile, Provenance(profile.context_hash), b"kv"
            )
        ledger.delete_document("tenant-a", "doc", expected_version=1)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0] == 0


def test_failed_transaction_rolls_back(ledger, document):
    with pytest.raises(RuntimeError, match="failure"):
        with ledger.transaction() as db:
            db.execute("DELETE FROM chunks")
            raise RuntimeError("failure")
    assert ledger.snapshot("tenant-a", "doc") == document


def test_empty_acl_denies_everyone(ledger, cache, alice, profile):
    doc = ledger.put_document("tenant-a", "doc", {"c1": "alpha"}, expected_version=0)
    assert cache.admit(doc.query("c1", alice), profile).status == Status.DENIED


def test_storage_failure_does_not_serve(document, alice, profile):
    ledger = Ledger()
    cache = VaultCache(ledger)
    ledger.close()
    with pytest.raises(sqlite3.ProgrammingError):
        cache.admit(document.query("c1", alice), profile)


@pytest.mark.parametrize("scope", [Scope.TENANT, Scope.PRIVATE, Scope.PUBLIC])
def test_cross_tenant_acl_rejected(ledger, scope):
    with pytest.raises(ValueError, match="workspace"):
        ledger.put_document(
            "tenant-a",
            "doc",
            {"c1": "alpha"},
            expected_version=0,
            acl_scope=scope,
            principals=[Identity("tenant-b", "alice")],
        )


def test_unchanged_policy_is_idempotent(ledger, document, alice, bob):
    assert (
        ledger.set_permissions(
            "tenant-a", "doc", expected_epoch=1, acl_scope=Scope.TENANT, principals=[alice, bob]
        )
        == document
    )


def test_missing_document_is_denied(cache, document, ledger, alice, profile):
    # A separate ledger has no knowledge of the document or the ticket.
    with Ledger() as empty:
        assert VaultCache(empty).admit(document.query("c1", alice), profile).status == Status.DENIED
