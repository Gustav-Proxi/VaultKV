import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from vaultkv import ConflictError
from vaultkv.txnrag import ArtifactKind, ReadMode, Resource, TxnRAG, TxnRejected
from vaultkv.txnrag.pipeline import prepare_answer


def texts(artifact):
    return [item["text"] for item in json.loads(artifact.payload)]


def test_pinned_snapshot_survives_update(store, token, resource, alice, profile):
    answer = prepare_answer(store, token, [resource])
    store.publish({resource: {"c1": "version-two"}}, expected_epoch=1)
    output = []
    store.release(token, answer, output.append)
    assert output == [b"version-one"]
    fresh = store.begin(alice, profile)
    assert fresh.knowledge_epoch == 2
    assert texts(store.retrieve(fresh, [resource])) == ["version-two"]


def test_multi_document_snapshot_cannot_tear(store, alice, profile, resource):
    second = Resource("tenant-a", "second")
    store.publish({resource: {"c": "A:old"}, second: {"c": "B:old"}}, expected_epoch=1)
    store.set_permissions(second, expected_epoch=1, principals=[alice])
    old = store.begin(alice, profile)
    first = store.retrieve(old, [resource])
    store.publish({resource: {"c": "A:new"}, second: {"c": "B:new"}}, expected_epoch=2)
    other = store.retrieve(old, [second])
    assert texts(first) + texts(other) == ["A:old", "B:old"]
    fresh = store.begin(alice, profile)
    assert texts(store.retrieve(fresh, [resource, second])) == ["A:new", "B:new"]


def test_freshest_mode_rejects_after_corpus_update(store, alice, profile, resource):
    token = store.begin(alice, profile, mode=ReadMode.FRESHEST)
    answer = prepare_answer(store, token, [resource])
    store.publish({resource: {"c1": "new"}}, expected_epoch=1)
    with pytest.raises(TxnRejected, match="knowledge_epoch_changed"):
        store.release(token, answer, lambda _: pytest.fail("Released obsolete freshest request"))
    store.cancel(token)


def test_delete_preserves_old_snapshot_but_revoke_blocks_it(store, token, resource, alice, profile):
    answer = prepare_answer(store, token, [resource])
    store.publish({}, expected_epoch=1, deletes=[resource])
    # Ordinary knowledge deletion is not an authorization revocation.
    assert texts(store.retrieve(token, [resource])) == ["version-one"]
    fresh = store.begin(alice, profile)
    with pytest.raises(TxnRejected, match="resource_not_in_snapshot"):
        store.retrieve(fresh, [resource])
    store.set_permissions(resource, expected_epoch=2)
    with pytest.raises(TxnRejected, match="permission_denied"):
        store.release(token, answer, lambda _: pytest.fail("Released revoked old snapshot"))


def test_delete_recreate_keeps_distinct_document_versions(store, token, resource, alice, profile):
    store.publish({}, expected_epoch=1, deletes=[resource])
    store.publish({resource: {"c1": "recreated"}}, expected_epoch=2)
    old = store.retrieve(token, [resource])
    new = store.retrieve(store.begin(alice, profile), [resource])
    assert old.dependencies[0].doc_version == 1
    assert new.dependencies[0].doc_version == 2
    assert texts(old) == ["version-one"]
    assert texts(new) == ["recreated"]


def test_knowledge_publication_never_restores_revoked_acl(store, resource, alice, profile):
    store.set_permissions(resource, expected_epoch=2)
    store.publish({resource: {"c1": "new"}}, expected_epoch=1)
    with pytest.raises(TxnRejected, match="permission_denied"):
        store.retrieve(store.begin(alice, profile), [resource])


def test_conflicting_publication_leaves_snapshot_unchanged(store, resource, token):
    with pytest.raises(ConflictError):
        store.publish({resource: {"c1": "conflicting"}}, expected_epoch=0)
    assert store.knowledge_epoch == 1
    assert texts(store.retrieve(token, [resource])) == ["version-one"]


def test_publication_rolls_back_all_resources_on_mid_write_failure(store, resource, monkeypatch):
    original = store._resource
    second = Resource("tenant-a", "second")

    def fail(db, selected):
        if selected == second:
            raise RuntimeError("injected publication failure")
        return original(db, selected)

    monkeypatch.setattr(store, "_resource", fail)
    with pytest.raises(RuntimeError, match="injected"):
        store.publish({resource: {"c1": "new"}, second: {"c": "new"}}, expected_epoch=1)
    assert store.knowledge_epoch == 1
    monkeypatch.setattr(store, "_resource", original)
    with pytest.raises(KeyError):
        store.resource_epoch(second)


def test_concurrent_publishers_have_exactly_one_winner(tmp_path, resource):
    path = tmp_path / "txn.sqlite3"
    with TxnRAG(path) as first, TxnRAG(path) as second:
        first.publish({resource: {"c": "initial"}}, expected_epoch=0)
        ready = Barrier(2)

        def publish(store, text):
            ready.wait(timeout=3)
            try:
                return store.publish({resource: {"c": text}}, expected_epoch=1)
            except ConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(publish, first, "first")
            b = pool.submit(publish, second, "second")
            assert {a.result(timeout=3), b.result(timeout=3)} == {2, "conflict"}


def test_cross_request_artifacts_cannot_be_mixed(store, token, resource, alice, profile):
    old = store.retrieve(token, [resource])
    store.publish({resource: {"c": "new"}}, expected_epoch=1)
    fresh_token = store.begin(alice, profile)
    new = store.retrieve(fresh_token, [resource])
    with pytest.raises(TxnRejected, match="cross_request_artifact"):
        store.derive(token, ArtifactKind.CONTEXT, [old, new], b"torn context")


def test_omitting_a_read_dependency_cannot_bypass_finality(store, token, resource, alice):
    second = Resource("tenant-a", "second")
    # Start a new request after publishing a second resource.
    store.publish({second: {"c": "another secret"}}, expected_epoch=1)
    store.set_permissions(second, expected_epoch=1, principals=[alice])
    token = store.begin(alice, token.profile)
    first = store.retrieve(token, [resource])
    store.retrieve(token, [second])
    answer = store.derive(token, ArtifactKind.ANSWER, [first], b"omitted dependency")
    with pytest.raises(TxnRejected, match="incomplete_answer_dependencies"):
        store.release(token, answer, lambda _: pytest.fail("Lost authorization closure"))


@pytest.mark.parametrize(
    "change", ["payload", "dependencies", "profile_fingerprint", "knowledge_epoch"]
)
def test_modified_artifact_is_rejected(store, token, resource, change):
    source = store.retrieve(token, [resource])
    changes = {
        "payload": b"poison",
        "dependencies": (),
        "profile_fingerprint": "wrong",
        "knowledge_epoch": 999,
    }
    with pytest.raises(TxnRejected):
        store.derive(
            token, ArtifactKind.ANSWER, [replace(source, **{change: changes[change]})], b"bad"
        )


def test_modified_token_cannot_choose_another_snapshot(store, token, resource):
    store.publish({resource: {"c1": "new"}}, expected_epoch=1)
    with pytest.raises(TxnRejected, match="unknown_or_modified_token"):
        store.retrieve(replace(token, knowledge_epoch=2), [resource])


def test_reopen_preserves_snapshots_tokens_and_artifacts(tmp_path, resource, alice, profile):
    path = tmp_path / "txn.sqlite3"
    with TxnRAG(path) as store:
        store.publish({resource: {"c": "old"}}, expected_epoch=0)
        store.set_permissions(resource, expected_epoch=1, principals=[alice])
        token = store.begin(alice, profile)
        answer = prepare_answer(store, token, [resource])
        store.publish({resource: {"c": "new"}}, expected_epoch=1)
    with TxnRAG(path) as store:
        output = []
        store.release(token, answer, output.append)
        assert output == [b"old"]
        assert store.knowledge_epoch == 2
