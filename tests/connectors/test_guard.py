from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from vaultkv.connectors.guard import NAMESPACE, PERMIT, TransferGuard
from vaultkv.txnrag import ArtifactKind, ReadMode, Resource, TxnRAG, TxnRejected


def issue(store, principal, profile, resource, ids=None, mode=ReadMode.SNAPSHOT):
    token = store.begin(principal, profile, mode=mode)
    evidence = store.retrieve(token, [resource])
    context = store.derive(token, ArtifactKind.CONTEXT, [evidence], b"same-context")
    return token, TransferGuard(store).issue(token, context, ids or [1, 2, 3])


def test_namespace_survives_unrelated_snapshot_but_not_source_edit(store, alice, profile, resource):
    _, first = issue(store, alice, profile, resource)
    store.publish({Resource("tenant-a", "other"): {"c": "different"}}, expected_epoch=1)
    _, second = issue(store, alice, profile, resource)
    assert first[NAMESPACE] == second[NAMESPACE]
    assert first[PERMIT] != second[PERMIT]
    store.publish({resource: {"c": "version-two"}}, expected_epoch=2)
    _, third = issue(store, alice, profile, resource)
    assert third[NAMESPACE] != second[NAMESPACE]


def test_identity_tokens_and_model_revision_isolate_cache(store, alice, bob, profile, resource):
    _, original = issue(store, alice, profile, resource)
    _, other_user = issue(store, bob, profile, resource)
    _, other_prompt = issue(store, alice, profile, resource, [1, 2, 4])
    _, other_model = issue(store, alice, replace(profile, model_revision="new"), resource)
    assert len({c[NAMESPACE] for c in [original, other_user, other_prompt, other_model]}) == 4


@pytest.mark.parametrize("mutation", ["revoke", "regrant", "principal", "policy", "closed"])
def test_current_authorization_is_required_at_every_use(
    store, request_context, resource, alice, mutation
):
    token, context = request_context
    guard = TransferGuard(store)
    configs = guard.issue(token, context, [1, 2, 3])
    if mutation in ("revoke", "regrant"):
        store.set_permissions(resource, expected_epoch=2)
        if mutation == "regrant":
            store.set_permissions(resource, expected_epoch=3, principals=[alice])
    elif mutation == "principal":
        store.set_principal_enabled(alice, expected_epoch=1, enabled=False)
    elif mutation == "policy":
        store.bump_policy_epoch(expected_epoch=1)
    else:
        store.cancel(token)
    with pytest.raises(TxnRejected), guard.checked([(configs, [1, 2, 3])]):
        pytest.fail("Backend must not execute")


@pytest.mark.parametrize("kind", ["unknown", "namespace", "missing", "override", "prompt"])
def test_forged_or_rebound_permits_fail(store, request_context, kind):
    token, context = request_context
    guard = TransferGuard(store)
    configs = guard.issue(token, context, [1, 2, 3])
    ids = [1, 2, 3]
    if kind == "unknown":
        configs[PERMIT] = "forged"
    elif kind == "namespace":
        configs[NAMESPACE] = "another-tenant"
    elif kind == "missing":
        del configs[PERMIT]
    elif kind == "override":
        configs["lmcache.skip_save"] = True
    else:
        ids = [1, 2, 4]
    with pytest.raises(TxnRejected), guard.checked([(configs, ids)]):
        pytest.fail("Forged transfer")


def test_partial_prefill_must_match_and_cannot_include_decode(store, request_context):
    token, context = request_context
    guard = TransferGuard(store)
    configs = guard.issue(token, context, [1, 2, 3])
    with guard.checked([(configs, [1, 2])], prefix=True):
        pass
    for ids in ([1, 9], [1, 2, 3, 4], []):
        with pytest.raises(TxnRejected), guard.checked([(configs, ids)], prefix=True):
            pytest.fail("Invalid worker prefix")
    guard.close(configs)
    with pytest.raises(TxnRejected), guard.checked([(configs, [1, 2, 3])]):
        pytest.fail("Closed permit")


def test_freshest_mode_rejects_publication(store, alice, profile, resource):
    _, configs = issue(store, alice, profile, resource, mode=ReadMode.FRESHEST)
    store.publish({resource: {"c": "edited"}}, expected_epoch=1)
    with pytest.raises(TxnRejected, match="knowledge_epoch_changed"):
        with TransferGuard(store).checked([(configs, [1, 2, 3])]):
            pytest.fail("Stale freshest request")


def test_read_set_cannot_grow_after_permit(store, resource, alice, profile):
    other = Resource("tenant-a", "extra")
    store.publish({other: {"c": "more evidence"}}, expected_epoch=1)
    store.set_permissions(other, expected_epoch=1, principals=[alice])
    token = store.begin(alice, profile)
    evidence = store.retrieve(token, [resource])
    context = store.derive(token, ArtifactKind.CONTEXT, [evidence], b"initial context")
    guard = TransferGuard(store)
    configs = guard.issue(token, context, [1, 2, 3])
    store.retrieve(token, [other])
    with pytest.raises(TxnRejected, match="connector_dependencies_changed"):
        with guard.checked([(configs, [1, 2, 3])]):
            pytest.fail("Permit no longer covers the read set")
    with pytest.raises(TxnRejected, match="incomplete_context_dependencies"):
        guard.issue(token, context, [1, 2, 3])


def test_transfer_holds_lock_until_synchronous_copy_finishes(
    store, request_context, resource, tmp_path
):
    token, context = request_context
    guard = TransferGuard(store)
    configs = guard.issue(token, context, [1, 2, 3])
    entered, finish, started, revoked = Event(), Event(), Event(), Event()
    with TxnRAG(tmp_path / "connector.sqlite3") as writer:

        def copy():
            with guard.checked([(configs, [1, 2, 3])]):
                entered.set()
                assert finish.wait(3)
                assert not revoked.is_set()

        def revoke():
            started.set()
            writer.set_permissions(resource, expected_epoch=2)
            revoked.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            transfer = pool.submit(copy)
            assert entered.wait(3)
            mutation = pool.submit(revoke)
            try:
                assert started.wait(3)
                assert not revoked.wait(0.05)
            finally:
                finish.set()
            transfer.result(3)
            mutation.result(3)
        with pytest.raises(TxnRejected), guard.checked([(configs, [1, 2, 3])]):
            pytest.fail("Revoked transfer")
