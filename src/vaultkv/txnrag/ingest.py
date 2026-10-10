"""Import UTF-8 Markdown/text files into versioned snapshots."""

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from vaultkv import Identity, content_hash
from vaultkv.txnrag import Resource, TxnRAG


def chunk_text(text: str, *, size: int = 900, overlap: int = 100) -> dict[str, str]:
    if size < 100 or not 0 <= overlap < size // 2:
        raise ValueError("Chunk size must be >= 100 and overlap less than half the size")
    text = text.strip()
    chunks, start = {}, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary != -1:
                end = boundary
        value = text[start:end].strip()
        if value:
            chunks[f"chunk-{len(chunks):04d}"] = value
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _current_chunk_hashes(store: TxnRAG, resource: Resource) -> dict[str, str] | None:
    """Chunk-id -> content hash of a resource in the current snapshot, or None if absent."""
    with store._ledger.transaction() as db:
        rows = db.execute(
            "SELECT c.chunk_id, c.content_hash FROM txn_meta m "
            "JOIN txn_snapshot_docs s ON s.epoch = m.knowledge_epoch "
            "JOIN txn_chunks c ON c.tenant_ns = s.tenant_ns AND c.doc_id = s.doc_id "
            "AND c.doc_version = s.doc_version WHERE s.tenant_ns = ? AND s.doc_id = ?",
            (resource.tenant_ns, resource.doc_id),
        ).fetchall()
    return {row[0]: row[1] for row in rows} or None


def ingest_documents(
    store: TxnRAG,
    documents: Mapping[str, str],
    principal: Identity,
    *,
    principals: tuple[Identity, ...] | None = None,
    groups: tuple[str, ...] = (),
) -> dict[str, object]:
    """Import named Markdown/text documents and grant access to new files."""
    updates, new = {}, []
    total_bytes = unchanged = 0
    for name, text in documents.items():
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError("Each document needs a filename and UTF-8 text")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or path.suffix.lower() not in (".md", ".txt")
        ):
            raise ValueError("Only safe .md and .txt filenames can be imported")
        size = len(text.encode("utf-8"))
        if size > 2_000_000:
            raise ValueError(f"File exceeds the 2 MB local prototype limit: {path.name}")
        total_bytes += size
        if total_bytes > 2_000_000:
            raise ValueError("Selected files exceed the 2 MB import limit")
        chunks = chunk_text(text)
        if not chunks:
            continue
        resource = Resource(principal.tenant_ns, path.as_posix())
        # Re-importing unchanged content must not publish new versions: that would bump
        # the knowledge epoch and invalidate every version-keyed cache on each refresh.
        if _current_chunk_hashes(store, resource) == {
            chunk_id: content_hash(chunk) for chunk_id, chunk in chunks.items()
        }:
            unchanged += 1
            continue
        updates[resource] = chunks
        try:
            store.resource_epoch(resource)
        except KeyError:
            new.append(resource)
    if not updates:
        if unchanged:
            return {
                "knowledge_epoch": store.knowledge_epoch,
                "documents": 0,
                "chunks": 0,
                "unchanged": unchanged,
                "new_grants": 0,
                "users": [],
                "groups": list(groups),
                "tenant": principal.tenant_ns,
            }
        raise ValueError("No nonempty UTF-8 .md or .txt files found")
    epoch = store.publish(updates, expected_epoch=store.knowledge_epoch)
    granted_users = principals if principals is not None else (principal,)
    for resource in new:
        store.set_permissions(
            resource,
            expected_epoch=1,
            principals=granted_users,
            groups=groups,
        )
    return {
        "knowledge_epoch": epoch,
        "documents": len(updates),
        "chunks": sum(len(chunks) for chunks in updates.values()),
        "unchanged": unchanged,
        "new_grants": len(new),
        "users": [identity.principal_id for identity in granted_users],
        "groups": list(groups),
        "tenant": principal.tenant_ns,
    }


def ingest_path(
    store: TxnRAG, path: Path, principal: Identity, *, name_root: Path | None = None
) -> dict[str, object]:
    """Grant only newly created resources; updates preserve current permissions.

    Missing files are NOT interpreted as deletes. Only .md/.txt files are imported.
    `name_root` keeps document names relative to a common root, so importing a large
    tree in several batches cannot collide (e.g. two folders that each hold index.md).
    """
    if not path.exists() or path.is_symlink():
        raise ValueError("Import path must be an existing nonsymlink file or directory")
    paths = [path] if path.is_file() else sorted(path.rglob("*"))
    root = path.parent if path.is_file() else path
    if name_root is not None:
        if not path.resolve().is_relative_to(name_root.resolve()):
            raise ValueError("Import path must be inside --name-root")
        root = name_root
    documents = {}
    for file in paths:
        if not file.is_file() or file.is_symlink() or file.suffix.lower() not in (".md", ".txt"):
            continue
        if file.stat().st_size > 2_000_000:
            raise ValueError(f"File exceeds the 2 MB local prototype limit: {file.name}")
        text = file.read_text(encoding="utf-8")
        if not text.strip():
            continue
        documents[file.resolve().relative_to(root.resolve()).as_posix()] = text
    return ingest_documents(store, documents, principal)
