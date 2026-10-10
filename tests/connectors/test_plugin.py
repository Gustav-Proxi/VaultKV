import importlib
import importlib.metadata
import sys
from types import ModuleType
from types import SimpleNamespace as Obj

import pytest

from vaultkv.connectors import LMCACHE_VERSION, VLLM_VERSION
from vaultkv.connectors.runtime import engine_config


@pytest.fixture
def plugin(monkeypatch):
    pins = {"vllm": VLLM_VERSION, "lmcache": LMCACHE_VERSION}
    monkeypatch.setattr(importlib.metadata, "version", pins.__getitem__)
    monkeypatch.setitem(sys.modules, "torch", Obj(cuda=Obj(synchronize=lambda: None)))
    for name in (
        "lmcache",
        "lmcache.integration",
        "lmcache.integration.vllm",
        "lmcache.integration.vllm.lmcache_connector_v1",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    class Upstream:
        def __init__(self, config, role):
            self._lmcache_engine = Obj(
                config=Obj(
                    **{
                        k.removeprefix("lmcache."): v
                        for k, v in config.kv_transfer_config.kv_connector_extra_config.items()
                        if k.startswith("lmcache.")
                    }
                )
            )

    leaf = sys.modules["lmcache.integration.vllm.lmcache_connector_v1"]
    leaf.LMCacheConnectorV1Dynamic = Upstream
    module = importlib.import_module("vaultkv.connectors.vllm")
    yield module.VaultKVLMCacheConnector
    sys.modules.pop("vaultkv.connectors.vllm", None)


def configuration(path):
    config = engine_config(path)
    return Obj(
        parallel_config=Obj(tensor_parallel_size=1, pipeline_parallel_size=1),
        cache_config=Obj(enable_prefix_caching=False, block_size=16),
        scheduler_config=Obj(enable_chunked_prefill=False, max_num_seqs=1),
        kv_transfer_config=Obj(**config["kv_transfer_config"]),
    )


def test_dynamic_constructor_accepts_vllm_three_argument_abi(plugin, tmp_path, store):
    instance = plugin(configuration(tmp_path / "connector.sqlite3"), "worker", Obj())
    instance._vault_store.close()


@pytest.mark.parametrize("bypass", ["apc", "parallel", "blend", "async", "disk"])
def test_plugin_rejects_unsupported_paths_before_engine_creation(plugin, tmp_path, store, bypass):
    config = configuration(tmp_path / "connector.sqlite3")
    extra = config.kv_transfer_config.kv_connector_extra_config
    if bypass == "apc":
        config.cache_config.enable_prefix_caching = True
    elif bypass == "parallel":
        config.parallel_config.tensor_parallel_size = 2
    else:
        key, value = {
            "blend": ("enable_blending", True),
            "async": ("enable_async_loading", True),
            "disk": ("local_disk", "/tmp/unsafe-cache"),
        }[bypass]
        extra["lmcache." + key] = value
    with pytest.raises(ValueError):
        plugin(config, "worker", None)
