"""VaultKV demos and local transactional retrieval/generation commands."""

import argparse
import json
from pathlib import Path

from vaultkv import CacheProfile, Identity, Ledger, Provenance, Rejected, Scope, VaultCache
from vaultkv.txnrag import Resource, TxnRAG


def demo() -> list[dict[str, str]]:
    alice = Identity("acme", "alice")
    bob = Identity("acme", "bob")
    outsider = Identity("other", "alice")
    profile = CacheProfile("demo-model@1", "demo-tokenizer@1", "demo-bytes-v1", "canonical-v1")
    events: list[dict[str, str]] = []
    with Ledger() as ledger:
        cache = VaultCache(ledger)
        doc = ledger.put_document(
            "acme",
            "handbook",
            {"intro": "Original policy"},
            expected_version=0,
            principals=[alice, bob],
        )
        query = doc.query("intro", alice)
        events.append({"event": "cold_lookup", "status": cache.admit(query, profile).status})
        ticket = cache.publish(query, profile, Provenance(profile.context_hash), b"synthetic-kv-v1")
        events.append({"event": "warm_lookup", "status": cache.admit(query, profile).status})
        events.append(
            {
                "event": "cross_tenant_lookup",
                "status": cache.admit(doc.query("intro", outsider), profile).status,
            }
        )
        doc = ledger.set_permissions(
            "acme",
            "handbook",
            expected_epoch=doc.policy_epoch,
            acl_scope=Scope.TENANT,
            principals=[bob],
        )
        try:
            cache.consume(ticket, lambda payload: len(payload))
        except Rejected as exc:
            events.append({"event": "queued_hit_after_revocation", "status": exc.decision.status})
        query = doc.query("intro", bob)
        ticket = cache.publish(query, profile, Provenance(profile.context_hash), b"synthetic-kv-v1")
        doc = ledger.put_document(
            "acme",
            "handbook",
            {"intro": "Revised policy"},
            expected_version=doc.doc_version,
            expected_epoch=doc.policy_epoch,
            principals=[bob],
        )
        try:
            cache.consume(ticket, lambda payload: len(payload))
        except Rejected as exc:
            events.append({"event": "queued_hit_after_edit", "status": exc.decision.status})
        query = doc.query("intro", bob)
        ledger.delete_document("acme", "handbook", expected_version=doc.doc_version)
        events.append(
            {"event": "lookup_after_delete", "status": cache.admit(query, profile).status}
        )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "txn-demo", "txn-races"):
        commands.add_parser(name)
    commands.add_parser("connector-doctor", help="Check the optional GPU connector prerequisites")
    connector = commands.add_parser("connector-config", help="Print pinned vLLM engine settings")
    connector.add_argument("--db", type=Path, default=Path("data/vaultkv.sqlite3"))
    serve = commands.add_parser("serve", help="Open the local operator dashboard")
    status = commands.add_parser("status", help="Inspect epochs, caches, and request counts")
    traces = commands.add_parser("traces", help="List persisted request traces")
    inspect = commands.add_parser("inspect", help="Inspect an authorized request trace")
    for command in (serve, status, traces, inspect):
        command.add_argument("--db", type=Path, default=Path("data/vaultkv.sqlite3"))
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true")
    api = commands.add_parser(
        "api", help="Cluster-mode API (/v1/ask, /healthz, /readyz, /metrics); token from env"
    )
    api.add_argument("--db", type=Path, default=Path("data/vaultkv.sqlite3"))
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)
    api.add_argument("--generator", choices=["ollama", "extractive"], default="ollama")
    api.add_argument("--model", default="qwen2.5:1.5b")
    api.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25")
    api.add_argument("--embedding-model", default="nomic-embed-text:latest")
    api.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    api.add_argument("--max-concurrent", type=int, default=2)
    inspect.add_argument("request_id")
    inspect.add_argument("--vectorlens", action="store_true")
    ingest = commands.add_parser("ingest", help="Import local .md/.txt files")
    ask = commands.add_parser("ask", help="Retrieve evidence and generate a buffered local answer")
    acl = commands.add_parser("set-acl", help="Replace a document's ACL (trusted administration)")
    for command in (ingest, ask, acl):
        command.add_argument("--db", type=Path, default=Path("data/vaultkv.sqlite3"))
        command.add_argument(
            "--workspace",
            "--tenant",
            dest="tenant",
            default="acme",
            metavar="WORKSPACE",
            help="Workspace name",
        )
    for command in (ingest, ask):
        command.add_argument(
            "--user",
            "--principal",
            dest="principal",
            default="alice",
            metavar="USER",
            help="User name",
        )
    ingest.add_argument("--path", type=Path, required=True)
    ingest.add_argument("--name-root", type=Path, help="Name documents relative to this directory")
    ask.add_argument("question")
    ask.add_argument("--model", default="qwen2.5:1.5b")
    ask.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    ask.add_argument("--model-revision", help="Immutable model commit for the vLLM backend")
    ask.add_argument("--embedding-model", default="nomic-embed-text:latest")
    ask.add_argument("--retrieval", choices=["bm25", "hybrid"], default="hybrid")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    acl.add_argument("--doc", required=True)
    acl.add_argument(
        "--allow-user",
        "--allow",
        dest="allow",
        action="append",
        default=[],
        metavar="USER",
        help="User allowed to access the document; repeat for more users",
    )
    acl.add_argument(
        "--scope", choices=[scope.value for scope in Scope], default=Scope.TENANT.value
    )
    args = parser.parse_args()
    if args.command in ("connector-doctor", "connector-config"):
        from vaultkv.connectors.runtime import doctor, engine_config

        try:
            result = doctor() if args.command == "connector-doctor" else engine_config(args.db)
            print(json.dumps(result, indent=2))
        except ValueError as exc:
            parser.exit(2, f"error: {exc}\n")
        return
    if args.command in ("serve", "status", "traces", "inspect"):
        from vaultkv.dashboard import serve as run_server
        from vaultkv.observability import inspect_trace, list_traces, snapshot, vectorlens_session

        try:
            if not args.db.is_file():
                raise ValueError("Database does not exist; ingest documents first")
            if args.command == "serve":
                run_server(args.db, port=args.port, open_browser=args.open)
            else:
                with TxnRAG(args.db) as store:
                    if args.command == "inspect":
                        inspect_trace(
                            store,
                            args.request_id,
                            lambda detail: print(
                                json.dumps(
                                    vectorlens_session(detail) if args.vectorlens else detail,
                                    indent=2,
                                ),
                                flush=True,
                            ),
                        )
                    else:
                        print(
                            json.dumps(
                                snapshot(store) if args.command == "status" else list_traces(store),
                                indent=2,
                            )
                        )
        except (ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"error: {exc}\n")
        return
    if args.command == "api":
        import os

        from vaultkv.api import ApiConfig, serve_api

        try:
            config = ApiConfig(
                db=args.db,
                token=os.environ.get("VAULTKV_API_TOKEN", ""),
                generator=args.generator,
                model=args.model,
                retrieval=args.retrieval,
                embedding_model=args.embedding_model,
                ollama_url=args.ollama_url,
                max_concurrent=args.max_concurrent,
            )
            serve_api(config, args.host, args.port)
        except (ValueError, OSError) as exc:
            parser.exit(2, f"error: {exc}\n")
        return
    if args.command in ("ingest", "ask", "set-acl"):
        try:
            _local_command(args)
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"error: {exc}\n")
        return
    if args.command == "demo":
        result = {"mode": "synthetic-cpu-demo", "events": demo()}
    elif args.command == "txn-demo":
        from vaultkv.txnrag.races import demo as txn_demo

        result = txn_demo()
    else:
        from vaultkv.txnrag.races import run_race_matrix

        result = run_race_matrix()
    print(json.dumps(result, indent=2))


def _local_command(args: argparse.Namespace) -> None:
    from vaultkv.txnrag.ingest import ingest_path
    from vaultkv.txnrag.local import LocalRAG
    from vaultkv.txnrag.ollama import OllamaClient

    if args.command != "ingest" and not args.db.is_file():
        raise ValueError("Database does not exist; ingest documents first")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with TxnRAG(args.db) as store:
        if args.command == "ingest":
            result = ingest_path(
                store, args.path, Identity(args.tenant, args.principal), name_root=args.name_root
            )
            print(json.dumps(result, indent=2))
        elif args.command == "set-acl":
            resource = Resource(args.tenant, args.doc)
            epoch = store.set_permissions(
                resource,
                expected_epoch=store.resource_epoch(resource),
                scope=Scope(args.scope),
                principals=[Identity(args.tenant, name) for name in args.allow],
            )
            print(json.dumps({"resource": args.doc, "authorization_epoch": epoch}))
        else:
            client = OllamaClient(args.ollama_url)
            if args.backend == "vllm":
                from vaultkv.connectors.client import VLLMClient

                if args.retrieval != "bm25" or not args.model_revision:
                    raise ValueError(
                        "vLLM currently requires --retrieval bm25 and --model-revision"
                    )
                client = VLLMClient(args.db, args.model, args.model_revision)
            rag = LocalRAG(
                store,
                client,
                model=args.model,
                embedding_model=args.embedding_model if args.retrieval == "hybrid" else None,
            )
            # Print the answer AND citations synchronously inside the authorization gate.
            rag.ask(
                Identity(args.tenant, args.principal),
                args.question,
                lambda payload: print(
                    json.dumps(json.loads(payload), indent=2, ensure_ascii=False), flush=True
                ),
                top_k=args.top_k,
            )


if __name__ == "__main__":
    main()
