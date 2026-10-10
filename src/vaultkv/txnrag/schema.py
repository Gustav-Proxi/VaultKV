"""Separate tables preserve the original VaultKV API and database semantics."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS txn_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    knowledge_epoch INTEGER NOT NULL,
    policy_epoch INTEGER NOT NULL
);
INSERT OR IGNORE INTO txn_meta VALUES (1, 0, 1);
CREATE TABLE IF NOT EXISTS txn_snapshots (epoch INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO txn_snapshots VALUES (0);
CREATE TABLE IF NOT EXISTS txn_resources (
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    latest_version INTEGER NOT NULL,
    auth_epoch INTEGER NOT NULL,
    scope TEXT NOT NULL,
    PRIMARY KEY (tenant_ns, doc_id)
);
CREATE TABLE IF NOT EXISTS txn_acl (
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    PRIMARY KEY (tenant_ns, doc_id, principal_id)
);
CREATE TABLE IF NOT EXISTS txn_principals (
    tenant_ns TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    auth_epoch INTEGER NOT NULL,
    enabled INTEGER NOT NULL,
    PRIMARY KEY (tenant_ns, principal_id)
);
CREATE TABLE IF NOT EXISTS txn_groups (
    tenant_ns TEXT NOT NULL,
    group_id TEXT NOT NULL,
    auth_epoch INTEGER NOT NULL,
    enabled INTEGER NOT NULL,
    PRIMARY KEY (tenant_ns, group_id)
);
CREATE TABLE IF NOT EXISTS txn_group_members (
    tenant_ns TEXT NOT NULL,
    group_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    PRIMARY KEY (tenant_ns, group_id, principal_id)
);
CREATE TABLE IF NOT EXISTS txn_acl_groups (
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    PRIMARY KEY (tenant_ns, doc_id, group_id)
);
CREATE TABLE IF NOT EXISTS txn_snapshot_docs (
    epoch INTEGER NOT NULL,
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_version INTEGER NOT NULL,
    PRIMARY KEY (epoch, tenant_ns, doc_id),
    FOREIGN KEY (epoch) REFERENCES txn_snapshots (epoch)
);
CREATE TABLE IF NOT EXISTS txn_chunks (
    tenant_ns TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_version INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (tenant_ns, doc_id, doc_version, chunk_id)
);
CREATE TABLE IF NOT EXISTS txn_requests (
    request_id TEXT PRIMARY KEY,
    token_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active', 'emitted', 'failed', 'cancelled'))
);
CREATE TABLE IF NOT EXISTS txn_dependencies (
    request_id TEXT NOT NULL,
    ref_json TEXT NOT NULL,
    PRIMARY KEY (request_id, ref_json),
    FOREIGN KEY (request_id) REFERENCES txn_requests (request_id)
);
CREATE TABLE IF NOT EXISTS txn_artifacts (
    artifact_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    descriptor_json TEXT NOT NULL,
    payload BLOB NOT NULL,
    FOREIGN KEY (request_id) REFERENCES txn_requests (request_id)
);
CREATE TABLE IF NOT EXISTS txn_cache (
    cache_key TEXT PRIMARY KEY,
    payload BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS txn_embeddings (
    cache_key TEXT PRIMARY KEY,
    vector_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS txn_answers (
    cache_key TEXT PRIMARY KEY,
    payload BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS txn_traces (
    request_id TEXT PRIMARY KEY REFERENCES txn_requests(request_id),
    created_at REAL NOT NULL,
    finished_at REAL,
    elapsed_ms REAL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS txn_connector_permits (
    permit TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES txn_requests(request_id),
    context_id TEXT NOT NULL REFERENCES txn_artifacts(artifact_id),
    namespace TEXT NOT NULL,
    token_ids_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS txn_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    request_id TEXT,
    kind TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS txn_events_request ON txn_events(request_id, id);
"""
