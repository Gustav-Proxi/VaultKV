"""Loopback-only operator dashboard with authenticated APIs and protected trace reads."""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from vaultkv import ConflictError, Identity, Scope
from vaultkv.observability import inspect_trace, list_traces, snapshot, vectorlens_session
from vaultkv.txnrag import Resource, TxnRAG, TxnRejected
from vaultkv.txnrag.ingest import ingest_documents
from vaultkv.txnrag.local import LocalRAG
from vaultkv.txnrag.ollama import ModelError, OllamaClient


class Dashboard:
    def __init__(self, path: Path, *, client_factory=OllamaClient) -> None:
        if not path.is_file():
            raise ValueError("Database does not exist; ingest documents first")
        self.path, self.client_factory = path, client_factory
        self.secret = secrets.token_urlsafe(32)
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.warmup = {"state": "starting"}
        self.pool.submit(self._warm_models)

    def _warm_models(self) -> None:
        try:
            client = self.client_factory()
            if hasattr(client, "warm"):
                embedding = client.resolve("nomic-embed-text:latest")
                client.embed(embedding, ["VaultKV model warmup"])
                client.warm(client.resolve("qwen2.5:1.5b"))
            with self.lock:
                self.warmup = {"state": "ready"}
        except Exception as exc:
            with self.lock:
                self.warmup = {
                    "state": "failed",
                    "error": str(exc) if isinstance(exc, ModelError) else type(exc).__name__,
                }

    def submit(self, body: dict) -> str:
        query = body.get("question", "")
        if not isinstance(query, str) or not query.strip() or len(query.encode()) > 1000:
            raise ValueError("Question must contain 1–1000 UTF-8 bytes")
        identity = Identity(body.get("tenant", "acme"), body.get("principal", "alice"))
        top_k = int(body.get("top_k", 3))
        if not 1 <= top_k <= 10 or body.get("retrieval", "smart") not in (
            "smart",
            "hybrid",
            "bm25",
        ):
            raise ValueError("Invalid retrieval settings")
        with self.lock:
            if sum(job["state"] in ("queued", "running") for job in self.jobs.values()) >= 4:
                raise ValueError("Four requests are already queued or running")
            for key in list(self.jobs):
                if len(self.jobs) <= 100:
                    break
                if self.jobs[key]["state"] not in ("queued", "running"):
                    del self.jobs[key]
            identifier = uuid4().hex
            self.jobs[identifier] = {"state": "queued", "request_id": None}

        def update(**values) -> None:
            with self.lock:
                self.jobs[identifier].update(values)

        def run() -> None:
            update(state="running")
            try:
                with TxnRAG(self.path) as store:
                    retrieval = body.get("retrieval", "smart")
                    rag = LocalRAG(
                        store,
                        self.client_factory(),
                        model=body.get("model", "qwen2.5:1.5b"),
                        embedding_model="nomic-embed-text:latest"
                        if retrieval in ("smart", "hybrid")
                        else None,
                        adaptive_retrieval=retrieval == "smart",
                    )
                    # Released output is captured transactionally by RequestTrace. GET
                    # rechecks current grants before any subsequent browser delivery.
                    rag.ask(
                        identity,
                        query,
                        lambda payload: None,
                        top_k=top_k,
                        on_start=lambda token: update(request_id=token.request_id),
                    )
                update(state="done")
            except Exception as exc:
                message = (
                    str(exc)
                    if isinstance(exc, (ModelError, TxnRejected, ValueError))
                    else type(exc).__name__
                )
                update(state="failed", error=message)

        self.pool.submit(run)
        return identifier


def make_server(app: Dashboard, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass  # Do not log user queries, credentials, or request bodies.

        def _send(self, payload: object, status: int = 200, mime: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def _trusted(self) -> bool:
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts:
                self._send({"error": "Invalid Host"}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in hosts}:
                self._send({"error": "Cross-origin access denied"}, 403)
                return False
            return True

        def _authorized(self) -> bool:
            supplied = self.headers.get("X-VaultKV-Token", "")
            if not hmac.compare_digest(supplied.encode(), app.secret.encode()):
                self._send({"error": "Open the dashboard URL printed by `vaultkv serve`"}, 401)
                return False
            return True

        def do_GET(self) -> None:
            if not self._trusted():
                return
            parsed = urlparse(self.path)
            assets = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if parsed.path in assets:
                name, mime = assets[parsed.path]
                self._send(files("vaultkv").joinpath("web", name).read_bytes(), mime=mime)
                return
            if not self._authorized():
                return
            try:
                if parsed.path == "/api/runtime":
                    try:
                        client = app.client_factory()
                        client.timeout = 3
                        models = client._request("/api/tags").get("models", [])
                        self._send(
                            {
                                "online": True,
                                "models": [
                                    {
                                        "name": m["name"],
                                        "digest": m["digest"],
                                        "size": m.get("size"),
                                    }
                                    for m in models
                                ],
                                "warmup": dict(app.warmup),
                            }
                        )
                    except ModelError:
                        self._send({"online": False, "models": [], "warmup": dict(app.warmup)})
                    return
                if parsed.path.startswith("/api/jobs/"):
                    with app.lock:
                        self._send(dict(app.jobs[parsed.path.rsplit("/", 1)[1]]))
                    return
                with TxnRAG(app.path) as store:
                    if parsed.path == "/api/overview":
                        self._send(snapshot(store))
                    elif parsed.path == "/api/traces":
                        self._send(list_traces(store))
                    elif parsed.path.startswith("/api/traces/"):
                        parts = parsed.path.split("/")
                        exporting = len(parts) == 5 and parts[4] == "vectorlens"
                        inspect_trace(
                            store,
                            parts[3],
                            lambda detail: self._send(
                                vectorlens_session(detail) if exporting else detail
                            ),
                        )
                    elif parsed.path == "/api/document":
                        query = parse_qs(parsed.query)
                        with store._ledger.transaction() as db:
                            meta = db.execute("SELECT knowledge_epoch FROM txn_meta").fetchone()[0]
                            tenant, doc = query["tenant"][0], query["doc"][0]
                            row = db.execute(
                                "SELECT * FROM txn_snapshot_docs WHERE epoch = ? "
                                "AND tenant_ns = ? AND doc_id = ?",
                                (meta, tenant, doc),
                            ).fetchone()
                            if row is None:
                                raise KeyError(doc)
                            chunks = [
                                dict(c)
                                for c in db.execute(
                                    "SELECT chunk_id, text, content_hash FROM txn_chunks "
                                    "WHERE tenant_ns = ? AND doc_id = ? AND doc_version = ? "
                                    "ORDER BY chunk_id",
                                    (tenant, doc, row["doc_version"]),
                                )
                            ]
                            self._send(
                                {
                                    "knowledge_epoch": meta,
                                    "chunks": chunks,
                                    "doc_version": row["doc_version"],
                                }
                            )
                    elif parsed.path == "/api/access":
                        query = parse_qs(parsed.query)
                        self._send(store.access_directory(query.get("tenant", ["acme"])[0]))
                    else:
                        self._send({"error": "Not found"}, 404)
            except KeyError:
                self._send({"error": "Not found"}, 404)
            except (ValueError, TxnRejected) as exc:
                self._send({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            if not self._trusted() or not self._authorized():
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < size <= 2_200_000
                    or self.headers.get("Content-Type", "").split(";")[0] != "application/json"
                ):
                    raise ValueError("Expected a JSON body under 2.2 MB")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/ask":
                    self._send({"job_id": app.submit(body)}, 202)
                    return
                if self.path == "/api/races":
                    from vaultkv.txnrag.races import run_race_matrix

                    self._send(run_race_matrix())
                    return
                with TxnRAG(app.path) as store:
                    if self.path == "/api/import":
                        files = body.get("files")
                        if not isinstance(files, list) or not 1 <= len(files) <= 20:
                            raise ValueError("Select between 1 and 20 documents")
                        documents = {}
                        for item in files:
                            if not isinstance(item, dict):
                                raise ValueError("Each document needs a filename and text")
                            name, text = item.get("name"), item.get("text")
                            if name in documents:
                                raise ValueError(f"Duplicate filename: {name}")
                            documents[name] = text
                        tenant = body.get("tenant", "acme")
                        users = body.get("principals", [body.get("principal", "alice")])
                        groups = body.get("groups", [])
                        if not isinstance(users, list) or not isinstance(groups, list):
                            raise ValueError("Users and groups must be lists")
                        result = ingest_documents(
                            store,
                            documents,
                            Identity(tenant, users[0] if users else "importer"),
                            principals=tuple(Identity(tenant, user) for user in users),
                            groups=tuple(groups),
                        )
                        self._send(result)
                        return
                    if self.path == "/api/users":
                        identity = Identity(body["tenant"], body["user"])
                        epoch = store.set_principal_enabled(
                            identity,
                            enabled=bool(body.get("enabled", True)),
                            expected_epoch=int(body.get("expected_epoch", 1)),
                        )
                        self._send({"access_version": epoch})
                        return
                    if self.path == "/api/groups":
                        members = body.get("members", [])
                        if not isinstance(members, list):
                            raise ValueError("Group members must be a list")
                        tenant = body["tenant"]
                        epoch = store.set_group(
                            tenant,
                            body["group"],
                            members=[Identity(tenant, user) for user in members],
                            enabled=bool(body.get("enabled", True)),
                            expected_epoch=int(body.get("expected_epoch", 0)),
                        )
                        self._send({"access_version": epoch})
                        return
                    resource = Resource(body["tenant"], body["doc"])
                    if self.path == "/api/permissions":
                        principals = body.get("principals", [])
                        groups = body.get("groups", [])
                        if not isinstance(principals, list) or not isinstance(groups, list):
                            raise ValueError("Users and groups must be lists")
                        epoch = store.set_permissions(
                            resource,
                            expected_epoch=int(body["expected_epoch"]),
                            scope=Scope(body["scope"]),
                            principals=[Identity(resource.tenant_ns, name) for name in principals],
                            groups=groups,
                        )
                        self._send({"authorization_epoch": epoch})
                    elif self.path == "/api/publish":
                        chunks = body["chunks"]
                        if (
                            not isinstance(chunks, dict)
                            or not chunks
                            or any(not isinstance(value, str) for value in chunks.values())
                        ):
                            raise ValueError("Expected nonempty chunk text mapping")
                        epoch = store.publish(
                            {resource: chunks}, expected_epoch=int(body["expected_epoch"])
                        )
                        self._send({"knowledge_epoch": epoch})
                    else:
                        self._send({"error": "Not found"}, 404)
            except ConflictError as exc:
                self._send({"error": str(exc)}, 409)
            except (ValueError, KeyError, TypeError, TxnRejected) as exc:
                self._send({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(path: Path, *, port: int = 8765, open_browser: bool = False) -> None:
    app = Dashboard(path)
    server = make_server(app, port)
    url = f"http://127.0.0.1:{server.server_port}/#access={app.secret}"
    print(f"VaultKV operator dashboard: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.pool.shutdown(wait=False, cancel_futures=True)
