from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vaultkv import Identity, Scope
from vaultkv.txnrag import TxnRAG, TxnRejected
from vaultkv.txnrag.pipeline import Phase, prepare_answer
from vaultkv.txnrag.races import run_case, run_race_matrix


@pytest.mark.parametrize("phase", list(Phase))
def test_every_revocation_barrier_blocks_output(phase):
    result = run_case("txnrag", phase)
    assert result["emissions"] == 0
    assert result["post_revocation_emissions"] == 0
    assert result["rejected"] == "permission_denied"


def test_baselines_demonstrate_missing_finality():
    assert run_race_matrix()["post_revocation_emissions"] == {
        "retrieval_only": 7,
        "retrieval_and_kv": 3,
        "txnrag": 0,
    }


@pytest.mark.parametrize("mutation", ["resource", "principal", "policy"])
def test_all_authorization_clocks_are_checked_before_output(
    store, token, resource, alice, mutation
):
    answer = prepare_answer(store, token, [resource])
    if mutation == "resource":
        # Even revoke/regrant cannot restore an old authorization generation.
        store.set_permissions(resource, expected_epoch=2)
        store.set_permissions(resource, expected_epoch=3, principals=[alice])
    elif mutation == "principal":
        store.set_principal_enabled(alice, enabled=False, expected_epoch=1)
        store.set_principal_enabled(alice, enabled=True, expected_epoch=2)
    else:
        store.bump_policy_epoch(expected_epoch=1)
    with pytest.raises(TxnRejected, match="changed"):
        store.release(token, answer, lambda _: pytest.fail("Old authorization accepted"))


def test_disabled_principal_cannot_begin_even_for_public_resource(store, resource, alice, profile):
    store.set_permissions(resource, expected_epoch=2, scope=Scope.PUBLIC)
    store.set_principal_enabled(alice, enabled=False, expected_epoch=1)
    with pytest.raises(TxnRejected, match="principal_disabled"):
        store.begin(alice, profile)


def test_same_principal_name_in_another_tenant_is_denied(store, resource, profile):
    token = store.begin(Identity("tenant-b", "alice"), profile)
    with pytest.raises(TxnRejected, match="permission_denied"):
        store.retrieve(token, [resource])


def test_generation_does_not_block_revocation_commit(tmp_path, resource, alice, profile):
    path = tmp_path / "concurrent.sqlite3"
    with TxnRAG(path) as reader, TxnRAG(path) as writer:
        reader.publish({resource: {"c": "secret"}}, expected_epoch=0)
        reader.set_permissions(resource, expected_epoch=1, principals=[alice])
        token = reader.begin(alice, profile)
        generating, resume = Event(), Event()
        output = []

        def barrier(phase):
            if phase == Phase.DURING_GENERATION:
                generating.set()
                assert resume.wait(timeout=3)

        def request():
            answer = prepare_answer(reader, token, [resource], barrier=barrier)
            reader.release(token, answer, output.append)

        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(request)
            try:
                assert generating.wait(timeout=3)
                mutation = pool.submit(writer.set_permissions, resource, expected_epoch=2)
                assert mutation.result(timeout=3) == 3  # Commits while generation is paused.
            finally:
                resume.set()
            with pytest.raises(TxnRejected, match="permission_denied"):
                future.result(timeout=3)
        assert output == []


def test_output_handoff_serializes_with_revocation(tmp_path, resource, alice, profile):
    path = tmp_path / "handoff.sqlite3"
    with TxnRAG(path) as reader, TxnRAG(path) as writer:
        reader.publish({resource: {"c": "secret"}}, expected_epoch=0)
        reader.set_permissions(resource, expected_epoch=1, principals=[alice])
        token = reader.begin(alice, profile)
        answer = prepare_answer(reader, token, [resource])
        emitting, finish, attempted, revoked = Event(), Event(), Event(), Event()
        output = []

        def sink(payload):
            emitting.set()
            assert finish.wait(timeout=3)
            assert not revoked.is_set()
            output.append(payload)

        def revoke():
            attempted.set()
            writer.set_permissions(resource, expected_epoch=2)
            revoked.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            response = pool.submit(reader.release, token, answer, sink)
            try:
                assert emitting.wait(timeout=3)
                mutation = pool.submit(revoke)
                assert attempted.wait(timeout=3)
                assert not revoked.wait(timeout=0.05)
            finally:
                finish.set()
            response.result(timeout=3)
            mutation.result(timeout=3)
        assert output == [b"secret"]  # Emitted before revocation became authoritative.
        assert revoked.is_set()


def test_sink_failure_closes_request_instead_of_replaying(store, token, resource):
    answer = prepare_answer(store, token, [resource])

    def broken_sink(payload):
        raise RuntimeError("transport failed after partial write")

    with pytest.raises(RuntimeError, match="transport failed"):
        store.release(token, answer, broken_sink)
    with pytest.raises(TxnRejected, match="request_closed"):
        store.release(token, answer, lambda _: pytest.fail("Replayed failed output"))


def test_successful_output_is_single_use(store, token, resource):
    answer = prepare_answer(store, token, [resource])
    output = []
    store.release(token, answer, output.append)
    with pytest.raises(TxnRejected, match="request_closed"):
        store.release(token, answer, output.append)
    assert output == [b"version-one"]


def test_cancelled_request_cannot_emit(store, token, resource):
    answer = prepare_answer(store, token, [resource])
    store.cancel(token)
    with pytest.raises(TxnRejected, match="request_closed"):
        store.release(token, answer, lambda _: pytest.fail("Emitted cancelled request"))


def test_async_sinks_are_rejected(store, token, resource):
    answer = prepare_answer(store, token, [resource])

    async def sink(payload):
        pytest.fail("Output deferred past the guard")

    with pytest.raises(TypeError, match="synchronous"):
        store.release(token, answer, sink)


def test_reentrant_sink_mutation_is_rejected(store, token, resource):
    answer = prepare_answer(store, token, [resource])
    with pytest.raises(RuntimeError, match="Reentrant"):
        store.release(token, answer, lambda _: store.set_permissions(resource, expected_epoch=2))
    assert store.resource_epoch(resource) == 2
