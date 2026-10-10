import pytest

from vaultkv.cache import VaultCache
from vaultkv.cli import demo
from vaultkv.workload import run_workload


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_independent_mutation_oracle(seed):
    result = run_workload(steps=500, seed=seed)
    assert result["unsafe_serves"] == 0
    assert result["counts"]["safe_consumptions"] > 0
    assert result["counts"]["queued_rejections"] > 0
    assert result["counts"]["edits"] > 0
    assert result["counts"]["acl_changes"] > 0
    assert result["counts"]["deletions"] > 0


def test_workload_is_deterministic():
    assert run_workload(steps=100, seed=1) == run_workload(steps=100, seed=1)


def test_oracle_detects_corrupted_payloads(monkeypatch):
    monkeypatch.setattr(VaultCache, "consume", lambda self, ticket, consumer: consumer(b"corrupt"))
    with pytest.raises(AssertionError, match="Unsafe serve"):
        run_workload(steps=100, seed=1)


def test_demo_outcomes():
    assert {row["event"]: row["status"] for row in demo()} == {
        "cold_lookup": "miss",
        "warm_lookup": "hit",
        "cross_tenant_lookup": "denied",
        "queued_hit_after_revocation": "denied",
        "queued_hit_after_edit": "stale",
        "lookup_after_delete": "denied",
    }
