import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vaultkv.observability import inspect_trace, list_traces, snapshot, vectorlens_session
from vaultkv.txnrag import TxnRAG, TxnRejected
from vaultkv.txnrag.local import LocalRAG
from vaultkv.txnrag.ollama import ModelRef


class Client:
    def resolve(self, name):
        return ModelRef(name, "fixed-digest")

    def generate(self, *args):
        self.last_generation_stats = {"prompt_eval_count": 42, "eval_count": 8}
        return "version-one [S1]"


def ask(store, alice, **kwargs):
    LocalRAG(store, Client(), embedding_model=None).ask(alice, "version", lambda _: None, **kwargs)
    return list_traces(store)[0]["request_id"]


def test_released_trace_and_vectorlens_export(store, alice):
    identifier = ask(store, alice)
    trace = inspect_trace(store, identifier, lambda data: data)
    assert trace["status"] == "emitted"
    assert trace["result"]["answer"] == "version-one [S1]"
    assert "result" not in list_traces(store)[0]
    assert trace["events"][-1]["kind"] == "finality.accepted"
    export = vectorlens_session(trace)
    assert export["llm_responses"][0]["prompt_tokens"] == 42
    assert export["vector_queries"][0]["results"][0]["text"] == "version-one"
    assert export["attributions"] == []
    assert export["vector_queries"][0]["db_type"] == "txnrag-sqlite-bm25"
    assert snapshot(store)["request_counts"] == {"emitted": 1}
    assert snapshot(store)["tensor_kv"]["hit_rate"] is None


@pytest.mark.parametrize("mutation", ["resource", "principal", "policy"])
def test_historical_content_is_reauthorized(store, alice, resource, mutation):
    identifier = ask(store, alice)
    if mutation == "resource":
        store.set_permissions(resource, expected_epoch=2)
    elif mutation == "principal":
        store.set_principal_enabled(alice, enabled=False, expected_epoch=1)
    else:
        store.bump_policy_epoch(expected_epoch=1)
    trace = inspect_trace(store, identifier, lambda data: data)
    assert trace["result"] is None
    assert trace["withheld"]
    export = vectorlens_session(trace)
    assert export["llm_responses"] == export["vector_queries"] == []
    assert "version-one" not in json.dumps(trace)


def test_blocked_trace_never_records_output(store, alice, resource):
    def revoke(phase, token):
        if phase == "after_generation":
            store.set_permissions(resource, expected_epoch=2)

    with pytest.raises(TxnRejected):
        ask(store, alice, barrier=revoke)
    trace = inspect_trace(store, list_traces(store)[0]["request_id"], lambda data: data)
    assert trace["status"] == "blocked"
    assert trace["result"] is None
    assert "version-one" not in json.dumps(trace)
    assert trace["events"][-1]["kind"] == "request.blocked"
    assert store._ledger._db.execute("SELECT result_json FROM txn_traces").fetchone()[0] is None


def test_async_output_cannot_be_marked_released(store, alice):
    async def output(payload):
        pytest.fail("Async output must never execute")

    with pytest.raises(TypeError, match="awaitables"):
        LocalRAG(store, Client(), embedding_model=None).ask(alice, "version", output)
    trace = inspect_trace(store, list_traces(store)[0]["request_id"], lambda data: data)
    assert trace["status"] == "failed"
    assert trace["result"] is None
    assert not any(e["kind"] == "finality.accepted" for e in trace["events"])


def test_inspection_handoff_serializes_with_revocation(tmp_path, resource, alice):
    path = tmp_path / "trace.sqlite3"
    with TxnRAG(path) as store, TxnRAG(path) as writer:
        store.publish({resource: {"c": "version-one"}}, expected_epoch=0)
        store.set_permissions(resource, expected_epoch=1, principals=[alice])
        identifier = ask(store, alice)
        entered, finish, started, revoked = Event(), Event(), Event(), Event()

        def sink(trace):
            assert trace["result"]
            entered.set()
            assert finish.wait(3)
            assert not revoked.is_set()

        def revoke():
            started.set()
            writer.set_permissions(resource, expected_epoch=2)
            revoked.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            reader = pool.submit(inspect_trace, store, identifier, sink)
            assert entered.wait(3)
            mutation = pool.submit(revoke)
            try:
                assert started.wait(3)
                assert not revoked.wait(0.05)
            finally:
                finish.set()
            reader.result(3)
            mutation.result(3)
        assert inspect_trace(store, identifier, lambda d: d)["result"] is None
