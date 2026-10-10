import pytest

from vaultkv import Identity
from vaultkv.txnrag import ArtifactKind, Resource, TxnRAG, TxnRejected


def test_group_grants_follow_membership_and_invalidate_active_reads(tmp_path, profile):
    path = tmp_path / "groups.sqlite3"
    resource = Resource("acme", "handbook.md")
    alice = Identity("acme", "alice")
    bob = Identity("acme", "bob")
    with TxnRAG(path) as store:
        store.publish({resource: {"section": "Internal handbook"}}, expected_epoch=0)
        assert store.set_group("acme", "staff", members=[alice], expected_epoch=0) == 1
        assert store.set_permissions(resource, expected_epoch=1, groups=["staff"]) == 2
        token = store.begin(alice, profile)
        evidence = store.retrieve(token, [resource])
        answer = store.derive(token, ArtifactKind.ANSWER, [evidence], b"answer")
        assert store.set_group("acme", "staff", members=[bob], expected_epoch=1) == 2
        with pytest.raises(TxnRejected, match="permission_denied"):
            store.release(token, answer, lambda _: None)
        with pytest.raises(TxnRejected, match="permission_denied"):
            store.retrieve(store.begin(alice, profile), [resource])
        assert store.retrieve(store.begin(bob, profile), [resource]).payload


def test_access_directory_lists_users_and_groups(tmp_path):
    with TxnRAG(tmp_path / "directory.sqlite3") as store:
        alice = Identity("acme", "alice")
        store.set_group("acme", "managers", members=[alice], expected_epoch=0)
        directory = store.access_directory("acme")
        assert directory["users"] == [{"user": "alice", "enabled": True, "access_version": 1}]
        assert directory["groups"] == [
            {
                "group": "managers",
                "enabled": True,
                "access_version": 1,
                "members": ["alice"],
            }
        ]
