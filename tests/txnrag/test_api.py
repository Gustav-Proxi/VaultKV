import json
import threading
from http.client import HTTPConnection

import pytest

from vaultkv import Identity
from vaultkv.api import ApiConfig, make_api_server
from vaultkv.txnrag import Resource, TxnRAG

TOKEN = "test-token-0123456789"


@pytest.fixture
def api(tmp_path):
    path = tmp_path / "api.sqlite3"
    with TxnRAG(path) as store:
        resource = Resource("acme", "travel")
        store.publish(
            {resource: {"c": "Hotels cost at most $180 per night in Buffalo."}}, expected_epoch=0
        )
        store.set_permissions(resource, expected_epoch=1, principals=[Identity("acme", "alice")])
    config = ApiConfig(db=path, token=TOKEN, generator="extractive")
    server = make_api_server(config, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def request(method, path, body=None, *, token=TOKEN):
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, data, headers)
        response = conn.getresponse()
        return response.status, response.read().decode()

    yield request, path
    server.shutdown()
    server.server_close()


def test_probes_and_metrics(api):
    request, _ = api
    assert request("GET", "/healthz") == (200, '{"status": "ok"}')
    status, body = request("GET", "/readyz")
    assert status == 200 and json.loads(body)["knowledge_epoch"] == 1
    status, body = request("GET", "/metrics")
    assert status == 200
    assert 'vaultkv_requests_total{status="emitted"} 0' in body
    assert "vaultkv_documents 1" in body


def test_ask_releases_cited_answer_and_counts_it(api):
    request, _ = api
    status, body = request(
        "POST",
        "/v1/ask",
        {"tenant": "acme", "principal": "alice", "question": "How much do Hotels cost per night?"},
    )
    payload = json.loads(body)
    assert status == 200
    assert payload["answer"].endswith("[S1]") and "$180" in payload["answer"]
    assert payload["request_id"]
    _, metrics = request("GET", "/metrics")
    assert 'vaultkv_requests_total{status="emitted"} 1' in metrics
    assert "vaultkv_request_duration_seconds_count 1" in metrics


def test_unauthorized_principal_gets_no_content(api):
    request, _ = api
    status, body = request(
        "POST",
        "/v1/ask",
        {
            "tenant": "acme",
            "principal": "mallory",
            "question": "How much do Hotels cost per night?",
        },
    )
    assert "$180" not in body
    assert status == 200 and "could not find supporting evidence" in json.loads(body)["answer"]


def test_revoked_principal_is_blocked_at_release(api, monkeypatch):
    request, path = api
    from vaultkv.txnrag.extractive import ExtractiveClient

    original = ExtractiveClient.generate

    def revoke_during_generation(self, model, system, prompt):
        with TxnRAG(path) as store:
            resource = Resource("acme", "travel")
            store.set_permissions(resource, expected_epoch=store.resource_epoch(resource))
        return original(self, model, system, prompt)

    monkeypatch.setattr(ExtractiveClient, "generate", revoke_during_generation)
    status, body = request(
        "POST",
        "/v1/ask",
        {"tenant": "acme", "principal": "alice", "question": "How much do Hotels cost per night?"},
    )
    assert status == 403 and "$180" not in body
    _, metrics = request("GET", "/metrics")
    assert 'vaultkv_requests_total{status="blocked"} 1' in metrics
    assert "vaultkv_blocked_requests_total{" in metrics


@pytest.mark.parametrize(
    ("token", "body", "expected"),
    [
        (None, {"tenant": "acme", "principal": "alice", "question": "q"}, 401),
        ("wrong-token-0123456789", {"tenant": "acme", "principal": "alice", "question": "q"}, 401),
        (TOKEN, {"tenant": "acme", "question": "q"}, 400),
        (TOKEN, {"tenant": "acme", "principal": "alice", "question": "q", "top_k": 99}, 400),
    ],
)
def test_rejects_bad_requests(api, token, body, expected):
    request, _ = api
    status, _ = request("POST", "/v1/ask", body, token=token)
    assert status == expected


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        ApiConfig(db=tmp_path / "x", token="short")
    with pytest.raises(ValueError):
        ApiConfig(db=tmp_path / "x", token=TOKEN, generator="extractive", retrieval="hybrid")


def test_serve_api_creates_empty_database(tmp_path, monkeypatch):
    import vaultkv.api as api_module

    started = {}

    class FakeServer:
        server_port = 0

        def serve_forever(self):
            started["db_exists"] = (tmp_path / "new" / "db.sqlite3").is_file()

        def server_close(self):
            pass

    monkeypatch.setattr(api_module, "make_api_server", lambda *args, **kwargs: FakeServer())
    config = ApiConfig(db=tmp_path / "new" / "db.sqlite3", token=TOKEN, generator="extractive")
    api_module.serve_api(config, "127.0.0.1", 0)
    assert started["db_exists"]
    with TxnRAG(config.db) as store:
        assert store.knowledge_epoch == 0
