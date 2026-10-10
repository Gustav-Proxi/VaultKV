"""Experimental dynamic vLLM V1 connector, pinned to the inspected upstream APIs.

This module intentionally requires GPU extras; the rest of VaultKV does not.
Only single-GPU, synchronous, local-CPU, prefix transfers are supported.
"""

from pathlib import Path

from vaultkv.connectors.runtime import require_versions

require_versions()

import torch  # noqa: E402
from lmcache.integration.vllm.lmcache_connector_v1 import (  # noqa: E402
    LMCacheConnectorV1Dynamic,
)

from vaultkv.connectors.hooks import GuardedHooks  # noqa: E402


class VaultKVLMCacheConnector(GuardedHooks, LMCacheConnectorV1Dynamic):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        config = vllm_config
        parallel = config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("VaultKV currently supports one GPU only")
        if (
            config.scheduler_config.enable_chunked_prefill
            or config.scheduler_config.max_num_seqs != 1
        ):
            raise ValueError("Disable chunked prefill and use one sequence for this prototype")
        if getattr(config, "lora_config", None) is not None:
            raise ValueError("LoRA adapters require additional provenance support")
        if config.cache_config.block_size != 16:
            raise ValueError("Use the pinned 16-token vLLM block size")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("Disable vLLM APC; it would bypass the external authorization gate")
        if config.kv_transfer_config.kv_role != "kv_both":
            raise ValueError("Disaggregated transfer roles are unsupported")
        extras = config.kv_transfer_config.kv_connector_extra_config
        required = {
            "lmcache.internal_api_server_enabled": False,
            "lmcache.extra_config": None,
            "lmcache.external_lookup_client": None,
            "lmcache.pre_caching_hash_algorithm": "sha256",
            "lmcache.chunk_size": 64,
            "lmcache.save_unfull_chunk": True,
            "lmcache.local_cpu": True,
            "lmcache.use_layerwise": False,
            "lmcache.enable_blending": False,
            "lmcache.enable_async_loading": False,
            "lmcache.save_decode_cache": False,
            "lmcache.enable_p2p": False,
            "lmcache.enable_pd": False,
            "lmcache.enable_controller": False,
            "lmcache.local_disk": None,
            "lmcache.remote_url": None,
        }
        if any(key not in extras or extras[key] != value for key, value in required.items()):
            raise ValueError("Use vaultkv connector-config; unsupported LMCache settings")
        path = Path(extras["vaultkv_db"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("All processes must share the same existing absolute database path")
        # This pinned upstream dynamic class has the old two-argument constructor.
        super().__init__(vllm_config, role)
        effective = self._lmcache_engine.config
        for key, value in required.items():
            if getattr(effective, key.removeprefix("lmcache.")) != value:
                raise ValueError("Effective LMCache configuration differs from guarded settings")
        self._init_guard(path, torch.cuda.synchronize)
