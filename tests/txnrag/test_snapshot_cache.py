from dataclasses import replace

import pytest

from vaultkv import Identity, Provenance, Scope
from vaultkv.txnrag import ArtifactKind, Resource, SnapshotCache, TxnRejected


def context(store, token, resource):
    evidence = store.retrieve(token, [resource])
    return store.derive(token, ArtifactKind.CONTEXT, [evidence], b"canonical-content")


def test_unchanged_sources_share_cache_across_knowledge_epochs(
    store, token, resource, alice, profile
):
    cache = SnapshotCache(store)
    first = context(store, token, resource)
    ticket = cache.publish(token, first, Provenance(profile.context_hash), b"kv")
    assert cache.load(token, ticket).payload == b"kv"
    store.publish({Resource("tenant-a", "unrelated"): {"c": "new"}}, expected_epoch=1)
    new = store.begin(alice, profile)
    hit = cache.admit(new, context(store, new, resource))
    assert hit is not None
    assert hit.cache_key == ticket.cache_key


def test_edited_sources_have_distinct_cache_keys(store, token, resource, alice, profile):
    cache = SnapshotCache(store)
    original = context(store, token, resource)
    ticket = cache.publish(token, original, Provenance(profile.context_hash), b"v1")
    store.publish({resource: {"c1": "version-two"}}, expected_epoch=1)
    new = store.begin(alice, profile)
    assert cache.admit(new, context(store, new, resource)) is None
    assert cache.load(token, ticket).payload == b"v1"  # Pinned K still valid.


@pytest.mark.parametrize(
    "field",
    ["model_revision", "tokenizer_revision", "kv_format", "context_hash", "adapter_revision"],
)
def test_computation_provenance_changes_miss(store, token, resource, alice, profile, field):
    cache = SnapshotCache(store)
    cache.publish(token, context(store, token, resource), Provenance(profile.context_hash), b"kv")
    new = store.begin(alice, replace(profile, **{field: "different"}))
    assert cache.admit(new, context(store, new, resource)) is None


@pytest.mark.parametrize("scope,should_share", [(Scope.TENANT, True), (Scope.PRIVATE, False)])
def test_sharing_respects_scope(store, resource, alice, bob, profile, scope, should_share):
    store.set_permissions(resource, expected_epoch=2, principals=[alice, bob], scope=scope)
    cache = SnapshotCache(store)
    first = store.begin(alice, profile)
    cache.publish(first, context(store, first, resource), Provenance(profile.context_hash), b"kv")
    second = store.begin(bob, profile)
    assert (cache.admit(second, context(store, second, resource)) is not None) == should_share


def test_public_sharing_across_tenants_stops_after_restriction(store, resource, alice, profile):
    store.set_permissions(resource, expected_epoch=2, scope=Scope.PUBLIC)
    cache = SnapshotCache(store)
    first = store.begin(alice, profile)
    cache.publish(
        first, context(store, first, resource), Provenance(profile.context_hash), b"public"
    )
    other = store.begin(Identity("tenant-b", "alice"), profile)
    ticket = cache.admit(other, context(store, other, resource))
    assert ticket is not None
    store.set_permissions(resource, expected_epoch=3, principals=[alice])
    with pytest.raises(TxnRejected, match="permission_denied"):
        cache.load(other, ticket)


def test_epoch_change_invalidates_authorized_sharing(store, token, resource, alice, bob, profile):
    cache = SnapshotCache(store)
    cache.publish(token, context(store, token, resource), Provenance(profile.context_hash), b"old")
    store.set_permissions(resource, expected_epoch=2, principals=[alice])
    new = store.begin(alice, profile)
    assert cache.admit(new, context(store, new, resource)) is None


def test_queued_lookup_rechecked_before_load(store, token, resource, profile):
    cache = SnapshotCache(store)
    ticket = cache.publish(
        token, context(store, token, resource), Provenance(profile.context_hash), b"kv"
    )
    store.set_permissions(resource, expected_epoch=2)
    with pytest.raises(TxnRejected, match="permission_denied"):
        cache.load(token, ticket)


def test_late_producer_cannot_publish_after_revocation(store, token, resource, profile):
    cache = SnapshotCache(store)
    source = context(store, token, resource)
    store.set_permissions(resource, expected_epoch=2)
    with pytest.raises(TxnRejected, match="permission_denied"):
        cache.publish(token, source, Provenance(profile.context_hash), b"late")


@pytest.mark.parametrize("provenance", [Provenance("poisoned"), Provenance("context@c", 8)])
def test_noncanonical_provenance_rejected(store, token, resource, provenance):
    with pytest.raises(ValueError, match="Canonical"):
        SnapshotCache(store).publish(token, context(store, token, resource), provenance, b"kv")


def test_ticket_cannot_substitute_another_context(store, token, resource, profile):
    cache = SnapshotCache(store)
    source = context(store, token, resource)
    ticket = cache.publish(token, source, Provenance(profile.context_hash), b"kv")
    with pytest.raises(TxnRejected, match="cache_ticket_mismatch"):
        cache.load(token, replace(ticket, cache_key="forged"))
