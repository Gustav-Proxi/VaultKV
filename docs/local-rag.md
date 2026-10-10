# Local retrieval and generation

The `ingest` and `ask` commands run real document retrieval and local LLM generation
through TxnRAG's snapshot and finality contracts. They use Python's standard library
plus a separately running local Ollama server.

## Quick start

```sh
uv sync --dev --locked
ollama serve  # Only if Ollama is not already running.
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
uv run vaultkv ingest --path examples/corpus --db data/local-demo.sqlite3
uv run vaultkv ask "What is the nightly hotel limit for domestic travel?" \
  --db data/local-demo.sqlite3 --top-k 2
```

The sample corpus contains invented travel, leave, and security policies. The
default workspace is `acme` and the default user is `alice`. Specify `--workspace`
and `--user` for another trusted local identity. The tested Qwen response was:

> The nightly hotel limit for domestic travel is $180 per night, excluding taxes [S1].

The JSON response includes source IDs, excerpts, document/chunk versions and
hashes, the knowledge epoch, and resolved generation/embedding model digests.
Answers can vary; one successful example is not an answer-quality benchmark.

## Import your files

```sh
uv run vaultkv ingest --path /path/to/documents --db data/my-corpus.sqlite3 \
  --workspace my-team --user alice
uv run vaultkv ask "What is the reimbursement deadline?" --db data/my-corpus.sqlite3 \
  --workspace my-team --user alice
```

Imports support UTF-8 `.md` and `.txt`, up to 2 MB per file. Chunks use a 900-character
window with 100-character overlap and word-boundary preference. IDs are relative
file paths plus sequential chunk IDs; boundaries are not stable after insertions.
Symlink files and unsupported file types are skipped. PDFs, Word, and HTML parsing
are not implemented.

All parsed content is published as one new knowledge snapshot. New resources are
then granted to the importing user; an interrupted access update can leave them
inaccessible until an administrator restores access. Reimporting an existing
resource preserves its current ACL, including revocations. Missing files are not
automatically deleted. Use a consistent import root to preserve resource IDs.

## Retrieval

`--retrieval hybrid` is the default. It combines:

- BM25 with Unicode word tokenization, `k1=1.2`, and `b=0.75`.
- Cosine similarity from Ollama embeddings, using Nomic's `search_document:` and
  `search_query:` prefixes.
- Reciprocal-rank fusion with constant 60 and a deterministic tie order.

Retrieval scans only resources present in the request's pinned snapshot and
authorized by the current ACL. Unauthorized chunk text is never loaded for
ranking or sent to the embedding model. Selected hits become registered evidence
with snapshot/version/hash and resource-epoch dependencies.

Document and query embeddings are built lazily and persisted in SQLite by source
identity or a one-way query digest, model digest, and preprocessing version.
Unchanged sources and repeated questions reuse vectors across snapshots. Model
calls occur outside database transactions. Authorization and snapshot compatibility
are checked again after ranking; concurrent permission changes abort the search
rather than return a result based on revoked inputs.

Use `--retrieval bm25` to skip embeddings. The final answer still uses the local
generation model. `--top-k` accepts 1–10, and `--embedding-model` selects an installed
embedding model; the current preprocessing is designed for Nomic. Dense search
is an exact in-memory scan, not an approximate nearest-neighbor index. The
implementation targets small local corpora and has no learned reranker.

The dashboard defaults to **Smart** retrieval. A clear keyword winner skips the
query-embedding call and sends only the matching evidence to generation. If the
keywords are ambiguous or absent, it automatically falls back to the same hybrid
meaning search described above. Choose **Thorough** to force vectors on every
question or **Fastest** for keyword-only retrieval.

## Generation and finality

The adapter uses Ollama's [`/api/embed`](https://docs.ollama.com/api/embed) and
[`/api/chat`](https://docs.ollama.com/api/chat) APIs. Model tags are resolved through
[`/api/tags`](https://docs.ollama.com/api/tags), and the digest is checked before
and after model calls. This detects changed tags in a trusted local deployment;
it is not cryptographic attestation of the executing weights.

Generation uses buffered chat (`stream: false`), temperature 0, seed 42, a
4096-token context setting, and up to 48 output tokens. The dashboard warms the
embedding and chat models in the background and keeps them loaded for 30 minutes,
avoiding the multi-second model-load cost on ordinary requests. Inputs and snippets
have conservative byte budgets. Embedding truncation is disabled; truncated or
invalid generation responses fail without releasing a partial answer. This bounds
local resource use but is not a tokenizer-level proof for every possible custom model.

An identical question over identical evidence can reuse its buffered answer. The
cache key binds the exact prompt, model revision, source versions, access epochs,
scope, and policy version. VaultKV repeats the current user and source access checks
at cache lookup and final output, so revocation still blocks a saved answer. Changed
content, access rules, model revisions, or prompts create a different key.

Only retrieved excerpts enter the prompt. Factual answers must include a source
citation; missing or nonexistent citation IDs are rejected. The returned source list identifies supplied evidence;
neither a valid citation nor this check proves that every generated claim is
supported. With no retrieved evidence, the application returns a fixed refusal
without calling the generation model.

The answer **and source excerpts** are one buffered artifact. Current source,
user, and policy access is checked before generation and again at the
synchronous output handoff. A revocation may commit while Ollama is computing; it
blocks subsequent output. Ordinary content updates preserve the pinned snapshot.
The Python API also supports `ReadMode.FRESHEST` to reject changed knowledge.

All HTTP requests are restricted to loopback, redirects are disabled, and HTTP
proxy environment variables are bypassed. `--ollama-url` can select another local
port. Ollama is a trusted internal processor: prompts may already have entered
its memory before a mid-request revocation. Revocation gates output; it does not
erase model memory, logs, or forensic data.

This adapter does not export or reuse KV tensors through VaultKV. `kv_reuse: false`
describes VaultKV-managed reuse; Ollama's own internal caching is not measured or
controlled here. A separate [experimental vLLM/LMCache backend](connectors.md)
implements guarded prefix transfers and awaits GPU validation.

## Permission administration

```sh
# Replace the travel policy ACL with Alice and Bob.
uv run vaultkv set-acl --db data/local-demo.sqlite3 --doc travel.md \
  --allow-user alice --allow-user bob

# Block everyone in the default workspace scope.
uv run vaultkv set-acl --db data/local-demo.sqlite3 --doc travel.md
```

`--scope` supports the internal values `tenant-shared` (listed users in the
workspace), `private` (one user), and `public` (everyone). These commands and the
workspace/user arguments are trusted local administration, not login/authentication.
Do not expose this CLI as an unrestricted multi-user service.

## Observability

Use `uv run vaultkv serve --db data/local-demo.sqlite3 --open` for live queries,
traces, cache/epoch activity and editing. See [dashboard documentation](dashboard.md)
for VectorLens exports, CLI inspection, and sensitive trace persistence.

## Verification

```sh
uv run pytest
uv run python benchmarks/local_rag_smoke.py
```

The ordinary suite uses deterministic model fixtures and a loopback HTTP test
server; no model downloads are required. The opt-in live smoke test requires both
models above. It asks about an invented secret, then uses two SQLite connections
to revoke access during a second real generation call and verifies that no answer
or citations escape. It uses a temporary database and leaves your corpus unchanged.

The local retrieval unit tests cover permission filtering before embedding,
producer-specific embedding reuse, content edits, mid-embedding revocation,
generation failure, source closure, and snapshot/freshest behavior during model
execution. Streaming and hardware-serving performance remain untested.
