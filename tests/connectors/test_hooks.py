from types import SimpleNamespace as Obj

import pytest

from vaultkv.connectors.guard import TransferGuard
from vaultkv.connectors.hooks import GuardedHooks
from vaultkv.txnrag import TxnRejected


class Mask:
    def __init__(self, count):
        self.count = count

    def sum(self):
        return self

    def item(self):
        return self.count


class Backend:
    def __init__(self, configs, count=3):
        self.calls = []
        self.metadata = Obj(
            requests=[Obj(req_id="engine-id", token_ids=[1, 2, 3], request_configs=configs)]
        )
        self._lmcache_engine = Obj(lmcache_engine=Obj(retrieve=lambda *a, **kw: Mask(count)))

    def _get_connector_metadata(self):
        return self.metadata

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.calls.append("lookup")
        return 2, False  # Last token recomputation remains the delegate's responsibility.

    def update_state_after_alloc(self, *args):
        self.calls.append("allocate")

    def build_connector_meta(self, *args):
        return self.metadata

    def start_load_kv(self, *args, **kwargs):
        self.calls.append("load")
        self._lmcache_engine.lmcache_engine.retrieve([1, 2, 3], Mask(3), req_id="engine-id")

    def save_kv_layer(self, *args, **kwargs):
        self.calls.append("layer")

    def wait_for_save(self):
        self.calls.append("save")


class Connector(GuardedHooks, Backend):
    pass


def make_connector(store, request_context, count=3):
    token, context = request_context
    configs = TransferGuard(store).issue(token, context, [1, 2, 3])
    connector = Connector(configs, count)
    connector._vault_store = store
    connector._vault_guard = TransferGuard(store)
    connector._synchronize = lambda: connector.calls.append("synchronize")
    request = Obj(
        prompt_token_ids=[1, 2, 3], sampling_params=Obj(extra_args={"kv_transfer_params": configs})
    )
    return connector, request


def test_pinned_hook_order_and_measured_load_counts(store, request_context):
    c, request = make_connector(store, request_context)
    assert c.get_num_new_matched_tokens(request, 0) == (2, False)
    c.update_state_after_alloc(request, None, 2)
    c.build_connector_meta(None)
    c.start_load_kv(None)
    c.save_kv_layer("layer", None, None)
    c.wait_for_save()
    assert c.calls == ["lookup", "allocate", "load", "synchronize", "layer", "save", "synchronize"]
    row = store._ledger._db.execute(
        "SELECT details_json FROM txn_events WHERE kind = 'connector.loaded'"
    ).fetchone()
    assert '"tokens": 3' in row[0]


@pytest.mark.parametrize("phase", ["lookup", "allocate", "metadata", "load", "layer", "save"])
def test_revocation_never_falls_back_to_a_miss(store, request_context, resource, phase):
    c, request = make_connector(store, request_context)
    if phase != "lookup":
        c.get_num_new_matched_tokens(request, 0)
    store.set_permissions(resource, expected_epoch=2)
    before = c.calls[:]
    calls = {
        "lookup": lambda: c.get_num_new_matched_tokens(request, 0),
        "allocate": lambda: c.update_state_after_alloc(request, None, 2),
        "metadata": lambda: c.build_connector_meta(None),
        "load": lambda: c.start_load_kv(None),
        "layer": lambda: c.save_kv_layer("layer", None, None),
        "save": c.wait_for_save,
    }
    with pytest.raises(TxnRejected, match="permission_denied"):
        calls[phase]()
    assert c.calls == before


def test_short_load_aborts_before_compute_and_records_no_success(store, request_context):
    c, _ = make_connector(store, request_context, count=2)
    original = c._lmcache_engine.lmcache_engine.retrieve
    with pytest.raises(TxnRejected, match="connector_incomplete_load"):
        c.start_load_kv(None)
    assert c._lmcache_engine.lmcache_engine.retrieve is original
    assert not store._ledger._db.execute(
        "SELECT 1 FROM txn_events WHERE kind = 'connector.loaded'"
    ).fetchall()
