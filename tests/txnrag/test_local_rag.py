import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vaultkv.txnrag import ReadMode, TxnRAG, TxnRejected
from vaultkv.txnrag.local import LocalRAG
from vaultkv.txnrag.ollama import ModelError, ModelRef


class Generator:
    def __init__(self):
        self.calls = []
        self.answer = "The answer is version-one [S1]."
        self.hook = lambda: None

    def resolve(self, model):
        return ModelRef(model, "digest-v1")

    def generate(self, model, system, prompt):
        self.calls.append(prompt)
        self.hook()
        return self.answer


def test_answer_and_citations_pass_through_gate(store, alice, resource):
    client = Generator()
    output = []
    LocalRAG(store, client, embedding_model=None).ask(alice, "version", output.append)
    result = json.loads(output[0])
    assert "version-one" in result["answer"]
    assert result["sources"][0]["source"]["resource"]["doc_id"] == resource.doc_id
    assert result["sources"][0]["id"] == "S1"
    assert result["kv_reuse"] is False
    assert result["answer_cache_hit"] is False


def test_identical_answer_is_reused_after_fresh_authorization_checks(store, alice, resource):
    client = Generator()
    output = []
    rag = LocalRAG(store, client, embedding_model=None)
    rag.ask(alice, "version", output.append)
    rag.ask(alice, "version", output.append)
    assert len(client.calls) == 1
    assert json.loads(output[1])["answer_cache_hit"] is True


def test_answer_cache_changes_with_question_and_content(store, alice, resource):
    client = Generator()
    rag = LocalRAG(store, client, embedding_model=None)
    rag.ask(alice, "version", lambda _: None)
    rag.ask(alice, "version one", lambda _: None)
    store.publish({resource: {"c": "version-two"}}, expected_epoch=1)
    rag.ask(alice, "version", lambda _: None)
    assert len(client.calls) == 3


def test_answer_cache_is_not_reused_after_access_changes(store, alice, resource):
    client = Generator()
    rag = LocalRAG(store, client, embedding_model=None)
    rag.ask(alice, "version", lambda _: None)
    store.set_permissions(resource, expected_epoch=2)
    store.set_permissions(resource, expected_epoch=3, principals=[alice])
    rag.ask(alice, "version", lambda _: None)
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "phase", ["after_retrieval", "before_generation", "after_generation", "before_release"]
)
def test_real_pipeline_gates_all_output_after_revocation(store, alice, resource, phase):
    client = Generator()
    output = []

    def revoke(point, token):
        if point == phase:
            store.set_permissions(resource, expected_epoch=2)

    with pytest.raises(TxnRejected, match="permission_denied"):
        LocalRAG(store, client, embedding_model=None).ask(
            alice, "version", output.append, barrier=revoke
        )
    assert output == []  # Neither text nor source excerpts escape.


def test_generation_releases_lock_and_revocation_prevents_output(tmp_path, resource, alice):
    with TxnRAG(tmp_path / "local.sqlite3") as store, TxnRAG(tmp_path / "local.sqlite3") as writer:
        store.publish({resource: {"c": "version-one"}}, expected_epoch=0)
        store.set_permissions(resource, expected_epoch=1, principals=[alice])
        client = Generator()
        started, finish = Event(), Event()

        def blocked_model():
            started.set()
            assert finish.wait(timeout=3)

        client.hook = blocked_model
        rag = LocalRAG(store, client, embedding_model=None)
        output = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            request = pool.submit(rag.ask, alice, "version", output.append)
            try:
                assert started.wait(timeout=3)
                mutation = pool.submit(writer.set_permissions, resource, expected_epoch=2)
                assert mutation.result(timeout=3) == 3
            finally:
                finish.set()
            with pytest.raises(TxnRejected, match="permission_denied"):
                request.result(timeout=3)
        assert output == []


@pytest.mark.parametrize("mode", [ReadMode.SNAPSHOT, ReadMode.FRESHEST])
def test_update_during_generation_obeys_read_mode(store, alice, resource, mode):
    client = Generator()
    client.hook = lambda: store.publish({resource: {"c": "version-two"}}, expected_epoch=1)
    rag = LocalRAG(store, client, embedding_model=None)
    output = []
    if mode == ReadMode.FRESHEST:
        with pytest.raises(TxnRejected, match="knowledge_epoch_changed"):
            rag.ask(alice, "version", output.append, mode=mode)
        assert output == []
    else:
        rag.ask(alice, "version", output.append, mode=mode)
        assert json.loads(output[0])["sources"][0]["excerpt"] == "version-one"


def test_no_evidence_does_not_call_model(store, alice):
    client = Generator()
    output = []
    LocalRAG(store, client, embedding_model=None).ask(alice, "unmatchedword", output.append)
    assert not client.calls
    assert json.loads(output[0])["sources"] == []


def test_unknown_citation_is_not_released(store, alice):
    client = Generator()
    client.answer = "Invented answer [S99]."
    with pytest.raises(ModelError, match="unknown source"):
        LocalRAG(store, client, embedding_model=None).ask(
            alice, "version", lambda _: pytest.fail("Leaked")
        )


def test_missing_citation_is_not_released(store, alice):
    client = Generator()
    client.answer = "An answer without evidence."
    with pytest.raises(ModelError, match="omitted its source citation"):
        LocalRAG(store, client, embedding_model=None).ask(
            alice, "version", lambda _: pytest.fail("Leaked")
        )


def test_model_failure_never_emits_sources(store, alice):
    client = Generator()

    def fail():
        raise ModelError("model failure")

    client.hook = fail
    with pytest.raises(ModelError):
        LocalRAG(store, client, embedding_model=None).ask(
            alice, "version", lambda _: pytest.fail("Leaked")
        )
