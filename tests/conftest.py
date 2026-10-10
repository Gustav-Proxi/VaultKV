import pytest

from vaultkv import CacheProfile, Identity, Ledger, VaultCache


@pytest.fixture
def ledger():
    with Ledger() as ledger:
        yield ledger


@pytest.fixture
def cache(ledger):
    return VaultCache(ledger)


@pytest.fixture
def alice():
    return Identity("tenant-a", "alice")


@pytest.fixture
def bob():
    return Identity("tenant-a", "bob")


@pytest.fixture
def profile():
    return CacheProfile("model@sha256:a", "tokenizer@sha256:b", "fp16:layout-v1", "context@c")


@pytest.fixture
def document(ledger, alice, bob):
    return ledger.put_document(
        "tenant-a",
        "doc",
        {"c1": "alpha", "c2": "beta"},
        expected_version=0,
        principals=[alice, bob],
    )
