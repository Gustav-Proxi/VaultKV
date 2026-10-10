"""Cluster-mode HTTP API: authenticated /v1/ask, health probes and /metrics.

Unlike the loopback dashboard, this binds to any interface for Kubernetes. The
bearer token represents a TRUSTED GATEWAY that has already authenticated the end
user and asserts their identity; never hand the token to end users. /metrics and
the probes are unauthenticated and expose only counts, epochs and durations.
"""

from __future__ import annotations

import hmac
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vaultkv import Identity
from vaultkv.metrics import render
from vaultkv.tracing import init_tracing, span
from vaultkv.txnrag import TxnRAG, TxnRejected
from vaultkv.txnrag.extractive import ExtractiveClient
from vaultkv.txnrag.local import LocalRAG
from vaultkv.txnrag.ollama import ModelError, OllamaClient

MAX_BODY_BYTES = 16 * 1024


@dataclass(frozen=True)
class ApiConfig:
    db: Path
    token: str
    generator: str = "ollama"  # "ollama" | "extractive"
    model: str = "qwen2.5:1.5b"
    retrieval: str = "bm25"  # "bm25" | "hybrid"
    embedding_model: str = "nomic-embed-text:latest"
    ollama_url: str = "http://127.0.0.1:11434"
    max_concurrent: int = 2

    def __post_init__(self) -> None:
        if len(self.token) < 16:
            raise ValueError("API token must be at least 16 characters")
        if self.generator not in ("ollama", "extractive"):
            raise ValueError("generator must be 'ollama' or 'extractive'")
        if self.retrieval not in ("bm25", "hybrid"):
            raise ValueError("retrieval must be 'bm25' or 'hybrid'")
        if self.retrieval == "hybrid" and self.generator == "extractive":
            raise ValueError("hybrid retrieval needs Ollama embeddings")

    def client(self):
        if self.generator == "extractive":
            return ExtractiveClient()
        return OllamaClient(self.ollama_url)


def _request_events(store: TxnRAG, request_id: str) -> list[tuple[float, str]]:
    with store._ledger.transaction() as db:
        return db.execute(
            "SELECT created_at, kind FROM txn_events WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()


def ask(config: ApiConfig, body: dict) -> tuple[int, dict]:
    question = body.get("question")
    tenant, principal = body.get("tenant"), body.get("principal")
    if not all(isinstance(v, str) and v for v in (question, tenant, principal)):
        raise ValueError("tenant, principal and question are required strings")
    top_k = body.get("top_k", 3)
    if not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10")
    released: list[bytes] = []
    request_ids: list[str] = []
    with span("vaultkv.ask", tenant=tenant, retrieval=config.retrieval) as current:
        with TxnRAG(config.db) as store:
            rag = LocalRAG(
                store,
                config.client(),
                model=config.model,
                embedding_model=config.embedding_model if config.retrieval == "hybrid" else None,
            )
            try:
                rag.ask(
                    Identity(tenant, principal),
                    question,
                    released.append,
                    top_k=top_k,
                    on_start=lambda token: request_ids.append(token.request_id),
                )
                status, payload = 200, json.loads(released[0])
            except TxnRejected as exc:
                status, payload = 403, {"error": "blocked", "reason": exc.reason}
            finally:
                if request_ids:
                    current.set_attribute("vaultkv.request_id", request_ids[0])
                    for created_at, kind in _request_events(store, request_ids[0]):
                        current.add_event(kind, timestamp=int(created_at * 1e9))
        current.set_attribute("http.status_code", status)
    if request_ids:
        payload["request_id"] = request_ids[0]
    return status, payload


def make_api_server(config: ApiConfig, host: str = "127.0.0.1", port: int = 8080):
    slots = threading.BoundedSemaphore(config.max_concurrent)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass  # Never log questions, identities or answers.

        def _send(self, status: int, payload: object, mime: str = "application/json") -> None:
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"status": "ok"})
            elif self.path == "/readyz":
                self._send(*self._ready())
            elif self.path == "/metrics":
                with TxnRAG(config.db) as store:
                    self._send(200, render(store), "text/plain; version=0.0.4")
            else:
                self._send(404, {"error": "not found"})

        def _ready(self) -> tuple[int, dict]:
            if not config.db.is_file():
                return 503, {"ready": False, "reason": "database missing; run ingest"}
            try:
                with TxnRAG(config.db) as store:
                    epoch = store.knowledge_epoch
                if config.generator == "ollama":
                    config.client().resolve(config.model)
            except ModelError as exc:
                return 503, {"ready": False, "reason": str(exc)}
            except Exception as exc:  # noqa: BLE001 - readiness must never crash the probe
                return 503, {"ready": False, "reason": type(exc).__name__}
            return 200, {"ready": True, "knowledge_epoch": epoch}

        def do_POST(self) -> None:
            if self.path != "/v1/ask":
                self._send(404, {"error": "not found"})
                return
            supplied = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(supplied.encode(), config.token.encode()):
                self._send(401, {"error": "invalid or missing bearer token"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                self._send(413 if length else 400, {"error": "body must be 1-16384 bytes"})
                return
            if not slots.acquire(blocking=False):
                self._send(429, {"error": "too many concurrent requests"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                self._send(*ask(config, body))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
            except ModelError as exc:
                self._send(502, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - never leak internals to callers
                self._send(500, {"error": type(exc).__name__})
            finally:
                slots.release()

    return ThreadingHTTPServer((host, port), Handler)


def serve_api(config: ApiConfig, host: str, port: int) -> None:
    init_tracing()
    # Start with an empty snapshot (epoch 0) if nothing was ingested yet, so the pod can
    # become ready before Helm's post-install ingest hook runs; answers simply find no
    # evidence until the first import publishes.
    config.db.parent.mkdir(parents=True, exist_ok=True)
    with TxnRAG(config.db):
        pass
    server = make_api_server(config, host, port)
    print(json.dumps({"listening": f"http://{host}:{server.server_port}"}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
