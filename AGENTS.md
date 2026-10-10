# Repository Guidelines

## Project Structure & Module Organization

- `src/vaultkv/` contains the ledger/cache API and CLI; `src/vaultkv/txnrag/` adds snapshots, retrieval, Ollama generation, and finality.
- `tests/` contains CPU unit, concurrency, persistence, and mutation-stream tests; `benchmarks/` contains the synthetic workload runner.
- `docs/` covers behavior and deployment; `examples/` contains public fixtures. Never commit caches, weights, databases, credentials, or private datasets.

## Build, Test, and Development Commands

Use Python 3.11+ and `uv`.

- `uv sync --dev --locked`: install the package and locked development dependencies.
- `uv run vaultkv demo`: demonstrate hits, revocation, and edits.
- `uv run pytest`: run automated tests.
- `uv run ruff check .` and `uv run ruff format --check .`: verify lint and formatting.
- `uv run vaultkv ingest --path examples/corpus`: import sample documents.
- `uv run vaultkv ask "What is the hotel limit?"`: retrieve and generate locally.
- `uv run vaultkv serve --db data/local-demo.sqlite3 --open`: launch the dashboard.
- `uv run python benchmarks/eval_gate.py`: eval gate run in CI (retrieval quality, permission leaks, finality, latency).
- `VAULTKV_API_TOKEN=... uv run vaultkv api --generator extractive`: cluster-mode API with `/metrics` and probes.
- `helm lint deploy/helm/vaultkv`: validate the Kubernetes chart; see `docs/deploy.md` for the cluster workflow.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and public-interface type annotations. Ruff enforces imports and a 100-character line limit; use `uv run ruff format .` before submitting.

## Testing Guidelines

Use pytest and `tests/test_*.py`; no numeric coverage threshold is configured. GPU checks use `benchmarks/connector_gpu_smoke.py`; see `docs/connectors.md`.

Cover invalidation, authorization epochs, workspace isolation, provenance, snapshots, and concurrent revocation. Denied requests must never recompute protected content. Output must finish synchronously inside `consume` or `release`; unsafe ablations belong only in the race harness.

## Local Agent Notes

Read `AGENTS.local.md` when it exists; it contains machine-specific workflow notes and is intentionally excluded from version control. Keep repository instructions portable and never commit private knowledge bases, credentials, model weights, or local databases.

## Commit & Pull Request Guidelines

No Git metadata is present; historical commit conventions are unknown. Use short imperative subjects.

PRs should explain the problem, affected components, validation, and relevant issue. Include workload settings for experiments; identify untested behavior and avoid presenting synthetic bytes as GPU KV results.
