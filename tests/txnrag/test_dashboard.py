import json
import threading
import time
from http.client import HTTPConnection

import pytest

from vaultkv import Identity
from vaultkv.dashboard import Dashboard, make_server
from vaultkv.txnrag import Resource, TxnRAG
from vaultkv.txnrag.ollama import ModelRef


class Client:
    def resolve(self, model):
        return ModelRef(model, "fixture")

    def generate(self, *args):
        return "Hotels cost $180 [S1]."

    def _request(self, *args):
        return {"models": [{"name": "fixture:latest", "digest": "fixture"}]}


@pytest.fixture
def dashboard(tmp_path):
    path = tmp_path / "test.sqlite3"
    with TxnRAG(path) as store:
        resource = Resource("acme", "travel")
        store.publish({resource: {"c": "Hotels cost $180."}}, expected_epoch=0)
        store.set_permissions(resource, expected_epoch=1, principals=[Identity("acme", "alice")])
    app = Dashboard(path, client_factory=Client)
    server = make_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, *, auth=True, headers=None):
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        values = {"X-VaultKV-Token": app.secret} if auth else {}
        if body is not None:
            values["Content-Type"] = "application/json"
        values.update(headers or {})
        conn.request(
            "POST" if body is not None else "GET",
            path,
            json.dumps(body) if body is not None else None,
            values,
        )
        response = conn.getresponse()
        data = response.read()
        result = (
            json.loads(data) if "application/json" in response.getheader("Content-Type") else data
        )
        conn.close()
        return response.status, result

    yield request, path
    server.shutdown()
    server.server_close()
    app.pool.shutdown(wait=True)
    thread.join(3)


def test_auth_origin_assets_and_runtime(dashboard):
    request, _ = dashboard
    assert request("/")[0] == 200
    assert b"VaultKV" in request("/")[1]
    assert request("/app.js")[0] == request("/style.css")[0] == 200
    assert request("/api/overview", auth=False)[0] == 401
    assert request("/api/overview", headers={"Origin": "https://untrusted.example"})[0] == 403
    assert request("/api/overview", headers={"Host": "attacker.example"})[0] == 403
    assert request("/api/runtime")[1]["models"][0]["name"] == "fixture:latest"
    assert request("/api/overview")[1]["knowledge_epoch"] == 1


def test_query_edit_revoke_and_export(dashboard):
    request, _ = dashboard
    code, response = request("/api/ask", {"question": "Hotels", "retrieval": "smart"})
    assert code == 202
    deadline = time.monotonic() + 3
    while True:
        job = request("/api/jobs/" + response["job_id"])[1]
        if job["state"] in ("done", "failed") or time.monotonic() > deadline:
            break
        time.sleep(0.01)
    assert job["state"] == "done", job
    endpoint = "/api/traces/" + job["request_id"]
    assert request(endpoint)[1]["result"]["answer"] == "Hotels cost $180 [S1]."
    assert request(endpoint + "/vectorlens")[1]["llm_responses"]
    body = {
        "tenant": "acme",
        "doc": "travel",
        "expected_epoch": 1,
        "chunks": {"c": "Hotels now cost $190."},
    }
    assert request("/api/publish", body)[1]["knowledge_epoch"] == 2
    assert request("/api/publish", body)[0] == 409
    assert (
        request("/api/document?tenant=acme&doc=travel")[1]["chunks"][0]["text"]
        == "Hotels now cost $190."
    )
    # Pinned historical content remains available until an authorization change.
    assert request(endpoint)[1]["result"]["sources"][0]["excerpt"] == "Hotels cost $180."
    code, _ = request(
        "/api/permissions",
        {
            "tenant": "acme",
            "doc": "travel",
            "expected_epoch": 2,
            "scope": "tenant-shared",
            "principals": [],
        },
    )
    assert code == 200
    assert request(endpoint)[1]["result"] is None
    assert request(endpoint + "/vectorlens")[1]["llm_responses"] == []


def test_invalid_query_and_document_input(dashboard):
    request, _ = dashboard
    assert request("/api/ask", {"question": ""})[0] == 400
    assert request("/api/ask", {"question": "x", "top_k": 99})[0] == 400
    assert (
        request(
            "/api/publish",
            {"tenant": "acme", "doc": "travel", "expected_epoch": 1, "chunks": {"c": 4}},
        )[0]
        == 400
    )
    assert request("/api/traces/missing")[0] == 404


def test_import_documents_from_dashboard(dashboard):
    request, _ = dashboard
    code, result = request(
        "/api/import",
        {
            "tenant": "acme",
            "principal": "bob",
            "files": [
                {"name": "benefits.md", "text": "Employees receive health coverage."},
                {"name": "contacts.txt", "text": "Support: help@example.test"},
            ],
        },
    )
    assert code == 200
    assert result["documents"] == result["new_grants"] == 2
    documents = {doc["doc_id"]: doc for doc in request("/api/overview")[1]["documents"]}
    assert documents["benefits.md"]["principals"] == ["bob"]
    assert documents["contacts.txt"]["principals"] == ["bob"]
    assert (
        request(
            "/api/import",
            {
                "tenant": "acme",
                "principal": "bob",
                "files": [{"name": "script.py", "text": "print('no')"}],
            },
        )[0]
        == 400
    )


def test_dashboard_users_groups_and_group_document_policy(dashboard):
    request, _ = dashboard
    assert (
        request(
            "/api/users",
            {
                "tenant": "acme",
                "user": "bob",
                "enabled": True,
                "expected_epoch": 1,
            },
        )[0]
        == 200
    )
    assert (
        request(
            "/api/groups",
            {
                "tenant": "acme",
                "group": "staff",
                "members": ["alice", "bob"],
                "enabled": True,
                "expected_epoch": 0,
            },
        )[0]
        == 200
    )
    directory = request("/api/access?tenant=acme")[1]
    assert [user["user"] for user in directory["users"]] == ["alice", "bob"]
    assert directory["groups"][0]["members"] == ["alice", "bob"]
    overview = request("/api/overview")[1]
    travel = next(doc for doc in overview["documents"] if doc["doc_id"] == "travel")
    assert (
        request(
            "/api/permissions",
            {
                "tenant": "acme",
                "doc": "travel",
                "expected_epoch": travel["auth_epoch"],
                "scope": "tenant-shared",
                "principals": [],
                "groups": ["staff"],
            },
        )[0]
        == 200
    )
    travel = next(
        doc for doc in request("/api/overview")[1]["documents"] if doc["doc_id"] == "travel"
    )
    assert travel["groups"] == ["staff"]
