# Deploying VaultKV (Kubernetes, monitoring, eval-gated CI)

## Cluster API

`vaultkv api` is the cluster-mode server. It binds any interface, unlike the loopback-only
`vaultkv serve` dashboard.

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/ask` | `Authorization: Bearer $VAULTKV_API_TOKEN` | `{"tenant","principal","question","top_k"}` → released answer + sources, or `403 {"error":"blocked","reason":…}` |
| `GET /healthz` | none | Liveness |
| `GET /readyz` | none | DB opens and, with `--generator ollama`, the model resolves |
| `GET /metrics` | none | Prometheus text: counts, epochs, durations only |

- **The token belongs to a trusted gateway.** The gateway authenticates the end user and asserts
  `tenant`/`principal`; never give the token to end users. Authorization (ACLs, revocation, the
  finality gate) is then enforced by TxnRAG on every request.
- **Generators:**
  - `--generator ollama` uses a local model through a loopback Ollama.
  - `--generator extractive` needs no model, for CI and smoke tests: answers are the top source's
    opening words plus a citation, still released through the finality gate.
- **Retrieval:** `--retrieval hybrid` (BM25 + Nomic embeddings) generalizes to paraphrased questions.
  `bm25` is keyword-only. Measured on `examples/eval`:

  | | lexical R@1 | paraphrase R@1 | paraphrase R@3 | paraphrase MRR |
  |---|---|---|---|---|
  | bm25 | 1.00 | 0.38 | 0.77 | 0.55 |
  | hybrid | 1.00 | 0.69 | 0.92 | 0.81 |

- **Tracing:** install the `otel` extra and set `OTEL_EXPORTER_OTLP_ENDPOINT` for OpenTelemetry spans.
  Each `vaultkv.ask` span carries the pipeline events (retrieval, context, generation, finality) as
  span events. Without it, tracing is a no-op.

## Metrics

| Metric | Type | Meaning |
|---|---|---|
| `vaultkv_requests_total{status}` | counter | emitted / blocked / failed / running |
| `vaultkv_blocked_requests_total{reason}` | counter | stopped by ACL or finality checks |
| `vaultkv_request_duration_seconds` | histogram | latency of released answers |
| `vaultkv_phase_elapsed_seconds{phase}` | histogram | time until retrieval / context / generation completed |
| `vaultkv_events_total{kind}` | counter | persisted pipeline events |
| `vaultkv_knowledge_epoch`, `vaultkv_policy_epoch` | gauge | snapshot and policy clocks (freshness) |
| `vaultkv_documents`, `vaultkv_chunks`, `vaultkv_principals{enabled}`, `vaultkv_cached_answers` | gauge | corpus and cache size |

Values are computed from the SQLite trace tables on each scrape, so they survive restarts.

## Helm chart (`deploy/helm/vaultkv`)

**Pod:**
- The API container runs non-root with a read-only root filesystem.
- An **Ollama sidecar** listens on `127.0.0.1`, which keeps the loopback-only client invariant.
- An init container pulls the models once onto their own volume.
- Probes use `/readyz` and `/healthz`.

**Storage:** SQLite on a ReadWriteOnce volume, so the Deployment runs exactly 1 replica with the
`Recreate` strategy.

**Ingest:**
- A post-install Job does the initial import, and a **CronJob** re-imports on a schedule. Any change
  publishes a new knowledge epoch, which is the freshness signal on the dashboard.
- `ingest.batches` splits large trees under the 2 MB-per-import cap. `--name-root` keeps document
  names unique across batches.
- **Ordering:** with `--wait`, Helm waits for the Deployment before running post-install hooks. The
  API therefore creates an empty database (epoch 0) at startup, becomes ready, and the ingest hook
  publishes afterwards.

**Monitoring (`monitoring.enabled`):**
- Prometheus scrapes `/metrics`, with alert rules: down, p95 > 30 s, blocked-request spike,
  failures, and no new snapshot in 24 h.
- Grafana has a provisioned **VaultKV** dashboard: epoch, documents, released vs blocked requests,
  p50/p95 latency, phase latency, blocked reasons and event rates.

**Pinned images:** `ollama/ollama:0.34.0`, `prom/prometheus:v3.14.0`, `grafana/grafana:13.0.5`;
`ghcr.io/astral-sh/uv:0.12.10` at build time.

## Local cluster serving a real knowledge base

```sh
brew install colima docker kind kubectl helm
colima start --cpu 4 --memory 8 --disk 40
cp deploy/helm/values-local.example.yaml deploy/helm/values-local.yaml
VAULT_DIR=~/knowledge-base MODE=full  scripts/dev-cluster.sh   # Ollama + hybrid retrieval
VAULT_DIR=~/knowledge-base MODE=light scripts/dev-cluster.sh   # extractive + BM25
```

- The ignored `deploy/helm/values-local.yaml` selects the tenant, user, folders and refresh schedule.
  The host directory is mounted read-only at `/vault`.
- The Service is ClusterIP only; use the port-forward commands printed in the chart NOTES.
- **Disk space:** full mode needs roughly 6–10 GB free (node image, ~1.3 GB of models, monitoring
  images). The VM's disk errors out if the Mac's disk fills.
- **Known issues:**
  - `kind load docker-image` can fail on newer Docker image stores, so the script loads a saved
    archive instead.
  - Remove a stale `"credsStore": "desktop"` from `~/.docker/config.json` if Docker Desktop was
    uninstalled.

## Eval-gated CI (`.github/workflows/ci.yml`)

1. **`test-and-eval`:**
   - Runs ruff, pytest, and `benchmarks/eval_gate.py` (BM25, deterministic).
   - The gate fails on any recall@1 / recall@3 / MRR drop over 0.02 vs
     `benchmarks/eval_baseline.json`, any restricted chunk retrieved by an unauthorized user, any
     post-revocation emission in the race matrix, or p95 retrieval latency over budget.
   - Accept intentional changes with `--update-baseline`. Run
     `benchmarks/eval_gate.py --retrieval hybrid` locally, with Ollama, for the hybrid numbers.
2. **`deploy-smoke`:**
   - Builds the image, lints the chart, creates a kind cluster, and installs the chart in extractive
     mode with monitoring.
   - Then checks readiness, an answer for an authorized user, **no content for an unauthorized user**,
     the metrics, and that Prometheus scrapes the API.

## Measured on a 16 GB M4 MacBook (colima: 4 vCPU / 8 GB, full mode)

Serving a private Markdown knowledge base: 180 documents, 2,859 chunks.

| Step | Time |
|---|---|
| First hybrid query (embeds every chunk once on CPU, ~4.4 chunks/s) | 11.6 min |
| Later questions (cached embeddings, Qwen 2.5 1.5B on CPU) | 5–6 s |
| Repeated questions (answer cache) | ~0.6 s |
| Re-import of unchanged vault | publishes nothing (epoch unchanged) |

- Disk: the colima VM file reached ~11–14 GB.
- **Quality:** retrieval returns the right documents, but the 1.5B model often answers "I do not know"
  or cites a source for a wrong fact. Use a larger model and/or a bigger excerpt budget before relying
  on answers. The eval gate measures retrieval only.

**Bugs found by running against the real vault (fixed, with tests):**
- Re-import published new versions of unchanged files every 15 minutes. That invalidated every
  version-keyed cache and grew the database.
- Embeddings were written only at the end of a request, so timeouts lost all progress and concurrent
  requests duplicated work. They are now persisted per 16-chunk batch.

Stop everything with `colima stop`. Resume with `colima start`; the cluster comes back.
