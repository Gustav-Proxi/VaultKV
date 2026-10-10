import importlib.metadata
import json
import subprocess
import sys

import pytest

from vaultkv.connectors import LMCACHE_VERSION, VLLM_VERSION
from vaultkv.connectors.runtime import doctor, engine_config, require_versions


def test_configuration_disables_bypasses_and_pins_model(tmp_path, store):
    settings = engine_config(tmp_path / "connector.sqlite3")
    assert len(settings["revision"]) == 40
    assert settings["tokenizer_revision"] == settings["revision"]
    assert settings["enable_prefix_caching"] is False
    assert settings["tensor_parallel_size"] == 1
    config = settings["kv_transfer_config"]
    assert config["kv_connector_module_path"] == "vaultkv.connectors.vllm"
    assert config["kv_connector_extra_config"]["lmcache.use_layerwise"] is False
    assert config["kv_connector_extra_config"]["lmcache.enable_blending"] is False
    with pytest.raises(ValueError, match="immutable"):
        engine_config(tmp_path / "connector.sqlite3", revision="main")
    with pytest.raises(ValueError):
        engine_config(tmp_path / "missing.sqlite3")


def test_version_mismatch_fails_instead_of_silent_fallback(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "wrong")
    with pytest.raises(RuntimeError, match="Unsupported vllm"):
        require_versions()
    pins = {"vllm": VLLM_VERSION, "lmcache": LMCACHE_VERSION}
    monkeypatch.setattr(importlib.metadata, "version", pins.__getitem__)
    require_versions()


def test_doctor_does_not_require_cuda_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vaultkv.connectors.runtime import doctor; "
            "import sys,json; print(json.dumps(doctor())); "
            "assert 'torch' not in sys.modules and 'vllm' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["stage"] == "experimental_prefix"
    assert doctor()["gpu_validated"] is False
