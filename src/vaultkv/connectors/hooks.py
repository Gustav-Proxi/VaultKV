"""Guarded vLLM V1 lifecycle hooks, testable without importing a GPU runtime."""

from pathlib import Path

from vaultkv.connectors.guard import TransferGuard
from vaultkv.txnrag import TxnRAG, TxnRejected
from vaultkv.txnrag.telemetry import record_event


class GuardedHooks:
    def _init_guard(self, path: Path, synchronize) -> None:
        self._vault_store = TxnRAG(path)
        self._vault_guard = TransferGuard(self._vault_store)
        self._synchronize = synchronize

    @staticmethod
    def _request_pair(request):
        if getattr(request, "mm_features", None):
            raise TxnRejected("connector_multimodal_unsupported")
        extra = request.sampling_params.extra_args or {}
        return extra.get("kv_transfer_params"), request.prompt_token_ids

    def _metadata_pairs(self):
        metadata = self._get_connector_metadata()
        return [(r.request_configs, r.token_ids) for r in metadata.requests]

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        with self._vault_guard.checked([self._request_pair(request)]) as (db, tokens):
            result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            matched, asynchronous = result
            if asynchronous or (matched is not None and (type(matched) is not int or matched < 0)):
                raise RuntimeError("Unsupported asynchronous or invalid lookup result")
            record_event(db, "connector.lookup", {"matched_tokens": matched}, tokens[0].request_id)
            return result

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        with self._vault_guard.checked([self._request_pair(request)]):
            return super().update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        with self._vault_guard.checked(
            [(r.request_configs, r.token_ids) for r in metadata.requests], prefix=True
        ):
            return metadata

    def start_load_kv(self, forward_context, **kwargs):
        pairs = self._metadata_pairs()
        with self._vault_guard.checked(pairs, prefix=True) as (db, tokens):
            # Upstream 0.3.9 only logs short loads. Reject them before computation,
            # and record actual returned token counts, not lookup predictions.
            engine = self._lmcache_engine.lmcache_engine
            original = engine.retrieve
            loaded = []

            def retrieve(*args, **values):
                result = original(*args, **values)
                expected = int(args[1].sum().item())
                actual = int(result.sum().item())
                if actual != expected:
                    raise TxnRejected("connector_incomplete_load")
                loaded.append((values.get("req_id"), actual))
                return result

            engine.retrieve = retrieve
            try:
                result = super().start_load_kv(forward_context, **kwargs)
            finally:
                try:
                    self._synchronize()
                finally:
                    engine.retrieve = original
            metadata = self._get_connector_metadata()
            by_id = {r.req_id: token for r, token in zip(metadata.requests, tokens, strict=True)}
            for request_id, count in loaded:
                if request_id not in by_id:
                    raise TxnRejected("connector_unknown_transfer")
                record_event(
                    db, "connector.loaded", {"tokens": count}, by_id[request_id].request_id
                )
            return result

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        # With layerwise disabled, LMCache stores in wait_for_save. Guard this
        # hook as well so switching upstream behavior cannot bypass the boundary.
        with self._vault_guard.checked(self._metadata_pairs(), prefix=True):
            return super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        with self._vault_guard.checked(self._metadata_pairs(), prefix=True) as (db, tokens):
            try:
                result = super().wait_for_save()
            finally:
                self._synchronize()
            for token in tokens:
                record_event(db, "connector.save_completed", {}, token.request_id)
            return result
