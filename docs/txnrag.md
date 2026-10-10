# TxnRAG local core

TxnRAG separates **knowledge snapshot consistency** from **authorization at output**.
The local implementation is in `src/vaultkv/txnrag/`. It uses VaultKV's SQLite
transaction plumbing and provenance models, with separate versioned tables and a
snapshot-aware cache adapter. The original `Ledger` and `VaultCache` API remains
available with its original strict invalidation semantics.

Real BM25/vector retrieval and local model generation are now available through
`LocalRAG` and the `ingest`/`ask` CLI. See [local RAG usage](local-rag.md). The
`prepare_answer` pipeline described below remains the synthetic reference harness.

## Run the demonstration

```sh
uv sync --dev --locked
uv run vaultkv txn-demo
uv run vaultkv txn-races
uv run pytest tests/txnrag -q
uv run python benchmarks/txnrag_races.py
```

The demo publishes K=1, prepares a response, publishes K=2, and releases the K=1
response under its pinned snapshot. A separate request loses access during
generation and emits nothing.

The race matrix injects a committed revocation at eight barriers:

1. Before retrieval.
2. After retrieval.
3. After reranking.
4. After context assembly.
5. After KV lookup but before loading.
6. After KV load.
7. During generation.
8. Immediately before response release.

For this fixed synthetic fixture, retrieval-only checks leak in 7 cases; retrieval
plus KV checks leak in 3; TxnRAG emits protected output in 0. These are controlled
software ablations, not evaluations of external products or real model attacks.
The detector uses the fault injector's completed-revocation flag and a known
secret in the emitted bytes; it does not ask the admission code whether output
was legal. Threaded tests additionally exercise independent database connections.

## Request lifecycle

```python
from vaultkv import CacheProfile, Identity
from vaultkv.txnrag import Resource, TxnRAG
from vaultkv.txnrag.pipeline import prepare_answer

alice = Identity("acme", "alice")
handbook = Resource("acme", "handbook")
profile = CacheProfile("model@revision", "tokenizer@revision", "bytes-v1", "canonical")

with TxnRAG() as store:
    store.publish({handbook: {"intro": "Travel allowance: 100"}}, expected_epoch=0)
    store.set_permissions(handbook, expected_epoch=1, principals=[alice])
    token = store.begin(alice, profile)
    answer = prepare_answer(store, token, [handbook])
    store.release(token, answer, lambda data: print(data.decode()))
```

`prepare_answer` is a keyed retriever, pass-through reranker, context assembler,
synthetic byte-cache client, and extractive response generator. There is no vector
index, BM25 ranker, learned reranker, or language model in this reference pipeline.
Its phase callback can coordinate fault injection without sleeping or holding a
database transaction during generation.

## Knowledge publication

`publish(updates, expected_epoch=K, deletes=...)` publishes an atomic corpus delta.
It copies the previous snapshot manifest, applies all changes, and advances K in
one transaction. Document versions and historical chunk texts are immutable.
New resources start with an empty workspace access list. Content publication never
changes existing access lists, so a stale content writer cannot restore a revoked grant.

A `TxnToken` pins the current K and records the user, user access version, policy
version, model profile, and read mode. Tokens are registered in the
database; modified or foreign tokens are rejected. Each evidence reference carries
K, resource/version/chunk identity, content hash, and the resource authorization
epoch captured when reading.

- **Snapshot mode** keeps using K even after later publications.
- **Freshest mode** rejects when current K changes; callers must begin again.
- **Deletion** removes a resource from new snapshots. Old snapshots may still use
  it if current authorization allows. For immediate non-disclosure, revoke access
  first with `set_permissions(resource, expected_epoch=..., principals=())`.

This is retained history, not secure erasure. Snapshot manifests are copied in
full; there is no compaction, retention policy, or garbage collector yet.

## Authorization and dependency closure

`set_permissions` changes a resource epoch; `set_principal_enabled` changes the
user access version; `bump_policy_epoch` invalidates requests after an external policy
change. Mutations require the expected epoch to prevent lost administrative
updates. These are trusted administrative operations, not an IAM provider.

Current authorization is checked before retrieval, cache admission/publication,
cache load, and output. Reranking/context/answer construction may finish privately
after revocation; private buffering is not permission to release.

Derived artifacts inherit the union of their registered parents' evidence.
Artifacts from another request/snapshot or computational profile cannot be mixed.
Payload or manifest tampering is rejected. The final answer must also contain the
entire request's read dependency set, preventing an omitted parent from bypassing
the final gate. This conservative rule may reject safe answers that used only a
subset of retrieved evidence. No semantic analysis proves which tokens depend on
which sources; processing code remains trusted.

Any resource-epoch change after evidence was read rejects that request, even if
the user remains authorized. A revoke/regrant cannot revive the old epoch.
Reusable workspace groups can be granted alongside individual users. Membership
or group-status changes advance every affected resource access version, so active
and historical reads are rechecked. Authorization-equivalence scopes and a general
policy language remain future work; the UI calls the current scopes Everyone,
Workspace, and This user.

## Cache semantics

`SnapshotCache` implements separate `admit`, `publish`, and `load` operations.
Queued tickets are revalidated before bytes are loaded. Its key binds source
versions/hashes, resource epochs, sharing scopes, context bytes, policy epoch, and
the model/tokenizer/adapter/context/format fingerprint. Private sharing also binds
the user. An unchanged source can be reused across knowledge epochs; an edit
creates a distinct key. Old snapshots can still use their compatible old entry.

The adapter stores opaque **whole-context bytes**. It does not implement tensor
fusion or chunk-position correction. The separate [experimental connector](connectors.md)
implements guarded prefix-transfer hooks; GPU execution is not yet validated.
Canonical-context and position-zero checks validate producer assertions; they do
not attest how opaque bytes were computed. Retired entries remain physically in
SQLite but are not reachable through authorized new requests. Query timing privacy
and provenance-attack resistance have not been established.

## Finality boundary and failures

`release(token, answer, sink)` revalidates the immutable contract, all source
permissions/epochs, and dependency closure in the same write transaction as the
synchronous sink handoff. Authorization becomes authoritative when its mutation
commits to this database. Revocation can commit during generation, but it waits for
an output handoff already holding the lock. Thus a local handoff either precedes a
revocation commit or checks the post-commit state and refuses output.

The sink must be short, synchronous, and non-reentrant. Do not retain bytes or
schedule async output beyond this boundary. The guarantee covers the handoff,
not later network delivery or when a person reads already emitted information.
No streaming lease, distributed commit, or cancellation of already delivered data
is implemented. Slow sinks delay revocation; SQLite has a five-second lock timeout
and propagates failures without allowing reuse.

Successful output closes the request. Sink exceptions also close it to avoid an
automatic duplicate response after a partial write. There is no atomic transaction
between SQLite and an external network sink: process crashes or commit failure
after a handoff require application-level delivery handling. `cancel(token)` closes
an unfinished request without output.

All Python callers, identities, producers, and database access are trusted. An
application must authenticate users and derive identities server-side; none of
these methods should be exposed as unrestricted client endpoints. Legacy VaultKV
and TxnRAG tables are independent: use TxnRAG's administration APIs to govern
TxnRAG requests, even when both APIs use the same database file.

## Remaining research work

Scalable retrieval indexes, GPU validation/non-prefix serving, ScopeKV group equivalence and
query privacy, general derived-state maintenance, streaming, snapshot reclamation,
and hardware performance evaluation remain future work. The local Ollama path
performs actual retrieval and generation without VaultKV-managed tensor reuse.
The supplied research report motivates this implementation; its literature and
novelty claims are not validated by these CPU tests.
