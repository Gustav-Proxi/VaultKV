"""Pinned experimental runtime configuration and dependency-free diagnostics."""

import importlib.metadata
import platform
import shutil
from pathlib import Path

from vaultkv.connectors import LMCACHE_VERSION, MODEL, MODEL_REVISION, VLLM_VERSION


def require_versions() -> None:
    for package, expected in (("vllm", VLLM_VERSION), ("lmcache", LMCACHE_VERSION)):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Install the GPU requirements first: missing {package}") from exc
        if actual != expected:
            raise RuntimeError(f"Unsupported {package} {actual}; connector requires {expected}")


def doctor() -> dict:
    packages = {}
    for name in ("vllm", "lmcache", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    reasons = []
    if platform.system() != "Linux":
        reasons.append("The experimental tensor connector targets Linux CUDA")
    if not shutil.which("nvidia-smi"):
        reasons.append("NVIDIA runtime was not detected (nvidia-smi missing)")
    for name, expected in (("vllm", VLLM_VERSION), ("lmcache", LMCACHE_VERSION)):
        if packages[name] != expected:
            reasons.append(f"Requires {name}=={expected}")
    return {
        "adapter": "VaultKVLMCacheConnector",
        "stage": "experimental_prefix",
        "platform": platform.system() + " " + platform.machine(),
        "packages": packages,
        "pins": {"vllm": VLLM_VERSION, "lmcache": LMCACHE_VERSION},
        "prerequisites_present": not reasons,
        "reasons": reasons,
        "gpu_validated": False,
        "tensor_metrics": "unavailable_until_worker_transfer",
    }


def engine_config(path: Path, model: str = MODEL, revision: str = MODEL_REVISION) -> dict:
    if not path.is_file():
        raise ValueError("Use an existing file-backed TxnRAG database")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Model revision must be an immutable 40-character commit SHA")
    return {
        "model": model,
        "revision": revision,
        "tokenizer_revision": revision,
        "dtype": "float16",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8192,
        "block_size": 16,
        "enforce_eager": True,
        "disable_hybrid_kv_cache_manager": True,
        "gpu_memory_utilization": 0.7,
        "kv_transfer_config": {
            "kv_connector": "VaultKVLMCacheConnector",
            "kv_connector_module_path": "vaultkv.connectors.vllm",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "vaultkv_db": str(path.resolve()),
                "lmcache.internal_api_server_enabled": False,
                "lmcache.extra_config": None,
                "lmcache.external_lookup_client": None,
                "lmcache.pre_caching_hash_algorithm": "sha256",
                "lmcache.local_cpu": True,
                "lmcache.max_local_cpu_size": 2.0,
                "lmcache.chunk_size": 64,
                "lmcache.save_unfull_chunk": True,
                "lmcache.use_layerwise": False,
                "lmcache.enable_blending": False,
                "lmcache.enable_async_loading": False,
                "lmcache.save_decode_cache": False,
                "lmcache.enable_p2p": False,
                "lmcache.enable_pd": False,
                "lmcache.enable_controller": False,
                "lmcache.local_disk": None,
                "lmcache.remote_url": None,
            },
        },
    }
