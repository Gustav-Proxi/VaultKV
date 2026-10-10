import pytest

from vaultkv.txnrag import Resource, TxnRAG


@pytest.fixture
def resource():
    return Resource("tenant-a", "doc")


@pytest.fixture
def store(resource, alice, bob):
    with TxnRAG() as store:
        store.publish({resource: {"c1": "version-one"}}, expected_epoch=0)
        store.set_permissions(resource, expected_epoch=1, principals=[alice, bob])
        yield store


@pytest.fixture
def token(store, alice, profile):
    return store.begin(alice, profile)
