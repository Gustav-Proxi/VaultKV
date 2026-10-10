import json

import pytest

from vaultkv import Identity
from vaultkv.txnrag import Resource, TxnRejected
from vaultkv.txnrag.ollama import ModelRef
from vaultkv.txnrag.retrieval import Retriever, bm25


class Embedder:
    def __init__(self):
        self.inputs = []
        self.hook = lambda: None

    def embed(self, model, texts):
        self.inputs.extend(texts)
        self.hook()
        # Deterministic semantic fixture: vehicle/automobile map to the same direction.
        return [
            [float(any(term in text for term in ("vehicle", "automobile"))), 1.0] for text in texts
        ]


def test_bm25_favors_matching_documents_and_handles_empty_input():
    scores = bm25("hotel reimbursement", ["hotel reimbursement limit 180", "annual leave policy"])
    assert scores[0] > scores[1] == 0
    assert bm25("query", []) == []
    assert bm25("query", ["", " "]) == [0, 0]


def test_retrieval_filters_before_embedding(store, token, resource):
    secret = Resource("tenant-a", "secret")
    store.publish({secret: {"c": "ULTRASECRET vehicle reimbursement"}}, expected_epoch=1)
    store.set_permissions(secret, expected_epoch=1, principals=[Identity("tenant-a", "eve")])
    fresh = store.begin(token.principal, token.profile)
    client = Embedder()
    result = Retriever(store, client=client, model=ModelRef("embed", "v1")).search(fresh, "vehicle")
    assert all("ULTRASECRET" not in value for value in client.inputs)
    assert all(ref.resource != secret for ref in result.dependencies)
    assert secret.doc_id not in result.payload.decode()


def test_dense_retrieval_adds_a_semantic_match(store, token, resource):
    store.publish({resource: {"c": "automobile"}}, expected_epoch=1)
    fresh = store.begin(token.principal, token.profile)
    assert json.loads(Retriever(store).search(fresh, "vehicle").payload) == []
    dense = Retriever(store, client=Embedder(), model=ModelRef("embed", "v1"))
    result = dense.search(fresh, "vehicle", top_k=1)
    assert json.loads(result.payload)[0]["text"] == "automobile"


def test_adaptive_retrieval_skips_dense_for_a_clear_keyword_match(store, token):
    client = Embedder()
    result = Retriever(store, client=client, model=ModelRef("embed", "v1"), adaptive=True).search(
        token, "version"
    )
    assert json.loads(result.payload)[0]["text"] == "version-one"
    assert client.inputs == []


def test_adaptive_retrieval_keeps_only_the_clear_keyword_winner(store, token, alice):
    policy = Resource("tenant-a", "hotel")
    distractor = Resource("tenant-a", "mention")
    store.publish(
        {
            policy: {"c": "nightly hotel reimbursement limit is 180 dollars"},
            distractor: {"c": "hotel"},
        },
        expected_epoch=1,
    )
    store.set_permissions(policy, expected_epoch=1, principals=[alice])
    store.set_permissions(distractor, expected_epoch=1, principals=[alice])
    fresh = store.begin(alice, token.profile)
    result = Retriever(
        store, client=Embedder(), model=ModelRef("embed", "v1"), adaptive=True
    ).search(fresh, "nightly hotel reimbursement limit", top_k=2)
    sources = json.loads(result.payload)
    assert len(sources) == 1
    assert sources[0]["source"]["resource"]["doc_id"] == "hotel"


def test_adaptive_retrieval_uses_meaning_fallback(store, token, resource):
    store.publish({resource: {"c": "automobile"}}, expected_epoch=1)
    fresh = store.begin(token.principal, token.profile)
    client = Embedder()
    result = Retriever(store, client=client, model=ModelRef("embed", "v1"), adaptive=True).search(
        fresh, "vehicle", top_k=1
    )
    assert json.loads(result.payload)[0]["text"] == "automobile"
    assert "search_query: vehicle" in client.inputs


def test_embedding_cache_reuses_only_matching_producer_and_content(store, token, resource):
    client = Embedder()
    retriever = Retriever(store, client=client, model=ModelRef("embed", "v1"))
    retriever.search(token, "version")
    client.inputs.clear()
    retriever.search(token, "version")
    assert client.inputs == []
    client.inputs.clear()
    changed = Retriever(store, client=client, model=ModelRef("embed", "v2"))
    changed.search(token, "version")
    assert any(text.startswith("search_document:") for text in client.inputs)
    store.publish({resource: {"c": "new text"}}, expected_epoch=1)
    fresh = store.begin(token.principal, token.profile)
    client.inputs.clear()
    retriever.search(fresh, "new")
    assert "search_document: new text" in client.inputs


def test_knowledge_update_during_embeddings_keeps_pinned_content(store, token, resource):
    client = Embedder()

    def publish():
        client.hook = lambda: None
        store.publish({resource: {"c1": "version-two"}}, expected_epoch=1)

    client.hook = publish
    result = Retriever(store, client=client, model=ModelRef("embed", "v1")).search(token, "version")
    assert json.loads(result.payload)[0]["text"] == "version-one"
    assert result.dependencies[0].knowledge_epoch == 1


def test_revocation_during_embedding_blocks_result(store, token, resource):
    client = Embedder()

    def revoke():
        client.hook = lambda: None
        store.set_permissions(resource, expected_epoch=2)

    client.hook = revoke
    with pytest.raises(TxnRejected, match="permission_denied"):
        Retriever(store, client=client, model=ModelRef("embed", "v1")).search(token, "version")


def test_search_pins_old_content_and_returns_only_selected_dependencies(store, token, resource):
    store.publish({resource: {"c": "new document"}}, expected_epoch=1)
    result = Retriever(store).search(token, "version")
    assert json.loads(result.payload)[0]["text"] == "version-one"
    assert len(result.dependencies) == 1


def test_empty_acl_returns_no_evidence_without_embedding_call(store, token, resource):
    store.set_permissions(resource, expected_epoch=2)
    client = Embedder()
    assert (
        json.loads(
            Retriever(store, client=client, model=ModelRef("e", "v1"))
            .search(token, "version")
            .payload
        )
        == []
    )
    assert client.inputs == []


def test_embedding_batches_persist_before_a_later_failure():
    import pytest

    from vaultkv import CacheProfile, Identity
    from vaultkv.txnrag import Resource, TxnRAG
    from vaultkv.txnrag.ollama import ModelError, ModelRef
    from vaultkv.txnrag.retrieval import Retriever

    class FlakyEmbedder:
        calls = 0

        def embed(self, model, texts):
            FlakyEmbedder.calls += 1
            if FlakyEmbedder.calls == 2:
                raise ModelError("embedding backend went away")
            return [[1.0, float(i + 1)] for i in range(len(texts))]

    owner = Identity("acme", "alice")
    resource = Resource("acme", "doc")
    with TxnRAG() as store:
        store.publish(
            {resource: {f"c{i:02d}": f"policy text number {i}" for i in range(20)}},
            expected_epoch=0,
        )
        store.set_permissions(resource, expected_epoch=1, principals=[owner])
        retriever = Retriever(store, client=FlakyEmbedder(), model=ModelRef("embed", "x"))
        token = store.begin(owner, CacheProfile("m", "m", "f", "c"))
        with pytest.raises(ModelError):
            retriever.search(token, "policy", top_k=3)
        store.cancel(token)
        with store._ledger.transaction() as db:
            cached = db.execute("SELECT count(*) FROM txn_embeddings").fetchone()[0]
        assert cached == 16  # the first batch survived the failure on the second
