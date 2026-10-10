import sys
from types import SimpleNamespace as Obj

import pytest

from vaultkv.connectors.client import VLLMClient, VLLMModelRef
from vaultkv.connectors.guard import TransferGuard
from vaultkv.observability import inspect_trace, list_traces
from vaultkv.txnrag import TxnRejected
from vaultkv.txnrag.local import LocalRAG
from vaultkv.txnrag.ollama import ModelError


def client(store, monkeypatch, *, finish_reason="stop", revoke=None):
    # No fake GPU counters: fixture exercises the producer contract and gate only.
    monkeypatch.setitem(sys.modules, "vllm", Obj(SamplingParams=lambda **kwargs: Obj(**kwargs)))
    adapter = VLLMClient.__new__(VLLMClient)
    adapter.model = VLLMModelRef("fixture", "a" * 40)
    adapter._bound = None
    adapter.last_generation_stats = {}

    def generate(prompt, params, **kwargs):
        with TransferGuard(store).checked(
            [(params.extra_args["kv_transfer_params"], prompt["prompt_token_ids"])]
        ):
            pass
        if revoke:
            revoke()
        return [
            Obj(
                finished=True,
                outputs=[Obj(text="version-one [S1]", token_ids=[9], finish_reason=finish_reason)],
            )
        ]

    adapter.engine = Obj(
        get_tokenizer=lambda: Obj(apply_chat_template=lambda *a, **kw: [1, 2, 3]),
        generate=generate,
    )
    return adapter


def test_vllm_producer_binds_context_and_releases_through_finality(store, alice, monkeypatch):
    adapter = client(store, monkeypatch)
    emitted = []
    LocalRAG(store, adapter, model="fixture", embedding_model=None).ask(
        alice, "version", emitted.append
    )
    trace = inspect_trace(store, list_traces(store)[0]["request_id"], lambda d: d)
    assert trace["result"]["generation_backend"] == "vllm-lmcache"
    assert trace["result"]["model_revision"].startswith("vllm:")
    assert trace["result"]["kv_reuse"] is False  # Fixture did not transfer tensors.
    assert len(emitted) == 1
    assert not store._ledger._db.execute("SELECT * FROM txn_connector_permits").fetchall()


def test_vllm_output_is_blocked_after_generation_revocation(store, resource, alice, monkeypatch):
    adapter = client(
        store, monkeypatch, revoke=lambda: store.set_permissions(resource, expected_epoch=2)
    )
    with pytest.raises(TxnRejected):
        LocalRAG(store, adapter, model="fixture", embedding_model=None).ask(
            alice, "version", lambda _: pytest.fail("Released revoked output")
        )
    assert list_traces(store)[0]["status"] == "blocked"
    assert not store._ledger._db.execute("SELECT * FROM txn_connector_permits").fetchall()


def test_truncated_generation_does_not_release_or_leave_permit(store, alice, monkeypatch):
    adapter = client(store, monkeypatch, finish_reason="length")
    with pytest.raises(ModelError):
        LocalRAG(store, adapter, model="fixture", embedding_model=None).ask(
            alice, "version", lambda _: pytest.fail("Released partial answer")
        )
    assert not store._ledger._db.execute("SELECT * FROM txn_connector_permits").fetchall()
