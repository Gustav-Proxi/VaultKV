from dataclasses import replace

import pytest

from vaultkv import Identity, Provenance, Rejected, Scope, Status


def test_cold_then_warm_hit(cache, document, alice, profile):
    query = document.query("c1", alice)
    assert cache.admit(query, profile).status == Status.MISS
    ticket = cache.publish(query, profile, Provenance(profile.context_hash), b"kv-alpha")
    assert cache.admit(query, profile).ticket == ticket
    assert cache.consume(ticket, lambda payload: payload.decode()) == "kv-alpha"


def test_nonprefix_chunk_lookup(cache, document, alice, profile):
    # Lookup c2 independently; c1 need not be cached or requested first.
    query = document.query("c2", alice)
    cache.publish(query, profile, Provenance(profile.context_hash), b"kv-beta")
    assert cache.admit(query, profile).status == Status.HIT
    assert cache.admit(document.query("c1", alice), profile).status == Status.MISS


@pytest.mark.parametrize("identity", [Identity("tenant-b", "alice"), Identity("tenant-a", "eve")])
def test_denied_cannot_publish_or_consume(cache, document, alice, profile, identity):
    ticket = cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"secret"
    )
    query = document.query("c1", identity)
    assert cache.admit(query, profile).status == Status.DENIED
    with pytest.raises(Rejected):
        cache.publish(query, profile, Provenance(profile.context_hash), b"poison")
    with pytest.raises(Rejected):
        cache.consume(replace(ticket, query=query), lambda _: pytest.fail("Loaded unauthorized KV"))


def test_authorized_tenant_sharing(cache, document, alice, bob, profile):
    cache.publish(document.query("c1", alice), profile, Provenance(profile.context_hash), b"shared")
    assert cache.admit(document.query("c1", bob), profile).status == Status.HIT


def test_private_cache_is_per_principal(ledger, cache, document, alice, bob, profile):
    document = ledger.set_permissions(
        "tenant-a", "doc", expected_epoch=1, acl_scope=Scope.PRIVATE, principals=[alice, bob]
    )
    cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"private"
    )
    assert cache.admit(document.query("c1", bob), profile).status == Status.MISS


def test_public_cache_can_be_shared_across_tenants(ledger, cache, document, alice, profile):
    document = ledger.set_permissions("tenant-a", "doc", expected_epoch=1, acl_scope=Scope.PUBLIC)
    ticket = cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"public"
    )
    decision = cache.admit(document.query("c1", Identity("tenant-b", "alice")), profile)
    assert decision.status == Status.HIT
    assert decision.ticket.cache_key == ticket.cache_key


@pytest.mark.parametrize(
    "field", ["model_revision", "tokenizer_revision", "kv_format", "context_hash"]
)
def test_profile_changes_are_cache_misses(cache, document, alice, profile, field):
    query = document.query("c1", alice)
    cache.publish(query, profile, Provenance(profile.context_hash), b"original")
    assert cache.admit(query, replace(profile, **{field: "different"})).status == Status.MISS


@pytest.mark.parametrize("provenance", [Provenance("poisoned-prefix"), Provenance("context@c", 12)])
def test_reject_noncanonical_provenance(cache, document, alice, profile, provenance):
    with pytest.raises(ValueError, match="canonical"):
        cache.publish(document.query("c1", alice), profile, provenance, b"poison")
    assert cache.admit(document.query("c1", alice), profile).status == Status.MISS


def test_first_publisher_wins(cache, document, alice, profile):
    query = document.query("c1", alice)
    ticket = cache.publish(query, profile, Provenance(profile.context_hash), b"first")
    cache.publish(query, profile, Provenance(profile.context_hash), b"second")
    assert cache.consume(ticket, lambda payload: payload.decode()) == "first"


def test_reject_tampered_ticket(cache, document, alice, profile):
    ticket = cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"data"
    )
    with pytest.raises(Rejected, match="ticket_mismatch"):
        cache.consume(
            replace(ticket, cache_key="forged"), lambda _: pytest.fail("Loaded forged KV")
        )


def test_reject_wrong_content_hash(cache, document, alice, profile):
    query = replace(document.query("c1", alice), content_hash="wrong")
    assert cache.admit(query, profile).status == Status.STALE
    with pytest.raises(Rejected, match="chunk_mismatch"):
        cache.publish(query, profile, Provenance(profile.context_hash), b"wrong")


def test_identical_document_ids_in_other_tenants_are_isolated(
    ledger, cache, document, alice, profile
):
    other = Identity("tenant-b", "alice")
    second = ledger.put_document(
        "tenant-b", "doc", {"c1": "alpha"}, expected_version=0, principals=[other]
    )
    cache.publish(
        document.query("c1", alice), profile, Provenance(profile.context_hash), b"original"
    )
    assert cache.admit(second.query("c1", other), profile).status == Status.MISS
