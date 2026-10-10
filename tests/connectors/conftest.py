import pytest

from vaultkv.txnrag import ArtifactKind, Resource, TxnRAG


@pytest.fixture
def resource():
    return Resource("tenant-a", "document")


@pytest.fixture
def store(tmp_path, resource, alice, bob):
    with TxnRAG(tmp_path / "connector.sqlite3") as store:
        store.publish({resource: {"c": "version-one"}}, expected_epoch=0)
        store.set_permissions(resource, expected_epoch=1, principals=[alice, bob])
        yield store


@pytest.fixture
def request_context(store, resource, alice, profile):
    token = store.begin(alice, profile)
    evidence = store.retrieve(token, [resource])
    context = store.derive(token, ArtifactKind.CONTEXT, [evidence], b"version-one")
    return token, context
