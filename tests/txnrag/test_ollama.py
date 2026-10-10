import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from vaultkv.txnrag.ollama import ModelError, ModelRef, OllamaClient


@pytest.fixture
def server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond({"models": [{"name": "test:latest", "digest": "digest1"}]})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, data))
            if self.path == "/api/chat":
                self.respond({"done": True, "message": {"content": "Grounded answer [S1]."}})
            elif self.path == "/api/generate":
                self.respond({"done": True})
            else:
                self.respond({"embeddings": [[1.0, 0.5] for _ in data["input"]]})

        def respond(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield OllamaClient(f"http://127.0.0.1:{httpd.server_port}"), requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_http_client_uses_buffered_chat_and_nontruncating_embeddings(server):
    client, requests = server
    model = client.resolve("test")
    assert client.embed(model, ["first", "second"]) == [[1.0, 0.5], [1.0, 0.5]]
    assert client.generate(model, "system", "question") == "Grounded answer [S1]."
    assert requests[0][1]["truncate"] is False
    assert requests[1][1]["stream"] is False
    assert requests[1][1]["options"]["num_ctx"] == 4096
    assert requests[1][1]["options"]["num_predict"] == 48


def test_model_warmup_loads_without_generating(server):
    client, requests = server
    client.warm(client.resolve("test"))
    assert requests[-1] == (
        "/api/generate",
        {
            "model": "test:latest",
            "prompt": "",
            "stream": False,
            "options": {"num_ctx": 4096},
            "keep_alive": "30m",
        },
    )


def test_model_revision_change_is_rejected(server):
    client, _ = server
    with pytest.raises(ModelError, match="revision changed"):
        client.verify(ModelRef("test:latest", "old-digest"))


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://10.0.0.1:11434",
        "http://user:pass@localhost",  # pragma: allowlist secret
        "http://localhost/api",
    ],
)
def test_client_refuses_nonlocal_or_ambiguous_urls(url):
    with pytest.raises(ValueError, match="loopback"):
        OllamaClient(url)


@pytest.mark.parametrize("vectors", [[], [[0.0, 0.0]], [[float("nan")]], [[True]], [[]]])
def test_invalid_embeddings_are_rejected(monkeypatch, vectors):
    client = OllamaClient()
    monkeypatch.setattr(client, "verify", lambda model: None)
    monkeypatch.setattr(client, "_request", lambda path, body: {"embeddings": vectors})
    with pytest.raises(ModelError):
        client.embed(ModelRef("test", "v1"), ["text"])


def test_truncated_generation_is_not_accepted(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "verify", lambda model: None)
    monkeypatch.setattr(
        client,
        "_request",
        lambda path, body: {
            "done": True,
            "done_reason": "length",
            "message": {"content": "partial"},
        },
    )
    with pytest.raises(ModelError, match="output budget"):
        client.generate(ModelRef("test", "v1"), "system", "question")
