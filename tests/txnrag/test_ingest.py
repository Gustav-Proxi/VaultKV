import json

import pytest

from vaultkv.txnrag import Resource, TxnRejected
from vaultkv.txnrag.ingest import chunk_text, ingest_path


def test_chunking_handles_long_words_and_unicode():
    chunks = chunk_text("🦊" * 2000, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 200 for chunk in chunks.values())
    assert list(chunks)[0] == "chunk-0000"
    assert chunk_text("   ") == {}


def test_import_update_preserves_acl(store, alice, profile, tmp_path):
    file = tmp_path / "notes.md"
    file.write_text("Original notes", encoding="utf-8")
    assert ingest_path(store, tmp_path, alice)["new_grants"] == 1
    resource = Resource(alice.tenant_ns, "notes.md")
    token = store.begin(alice, profile)
    assert json.loads(store.retrieve(token, [resource]).payload)[0]["text"] == "Original notes"
    store.set_permissions(resource, expected_epoch=2)
    file.write_text("Updated notes", encoding="utf-8")
    assert ingest_path(store, tmp_path, alice)["new_grants"] == 0
    with pytest.raises(TxnRejected, match="permission_denied"):
        store.retrieve(store.begin(alice, profile), [resource])


def test_bad_import_does_not_publish_partial_corpus(store, alice, tmp_path):
    (tmp_path / "a.md").write_text("Good file", encoding="utf-8")
    (tmp_path / "b.txt").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(UnicodeDecodeError):
        ingest_path(store, tmp_path, alice)
    assert store.knowledge_epoch == 1


def test_symlinks_and_unsupported_files_are_not_imported(store, alice, tmp_path):
    (tmp_path / "a.md").write_text("Sample", encoding="utf-8")
    (tmp_path / "b.md").symlink_to(tmp_path / "a.md")
    (tmp_path / "script.py").write_text("ignored", encoding="utf-8")
    assert ingest_path(store, tmp_path, alice)["documents"] == 1


def test_name_root_prevents_batch_collisions(tmp_path):
    from vaultkv import Identity
    from vaultkv.txnrag import Resource, TxnRAG
    from vaultkv.txnrag.ingest import ingest_path

    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "index.md").write_text(f"notes for {folder}")
    owner = Identity("vault", "me")
    with TxnRAG() as store:
        ingest_path(store, tmp_path / "a", owner, name_root=tmp_path)
        ingest_path(store, tmp_path / "b", owner, name_root=tmp_path)
        assert store.resource_epoch(Resource("vault", "a/index.md"))
        assert store.resource_epoch(Resource("vault", "b/index.md"))
    with TxnRAG() as store, pytest.raises(ValueError):
        ingest_path(store, tmp_path / "a", owner, name_root=tmp_path / "b")


def test_reimporting_unchanged_content_publishes_nothing(tmp_path):
    from vaultkv import Identity
    from vaultkv.txnrag import TxnRAG
    from vaultkv.txnrag.ingest import ingest_path

    (tmp_path / "a.md").write_text("alpha notes " * 20)
    (tmp_path / "b.md").write_text("beta notes " * 20)
    owner = Identity("vault", "me")
    with TxnRAG() as store:
        first = ingest_path(store, tmp_path, owner)
        assert first["knowledge_epoch"] == 1 and first["documents"] == 2
        again = ingest_path(store, tmp_path, owner)
        assert again["knowledge_epoch"] == 1 and again["documents"] == 0
        assert again["unchanged"] == 2
        (tmp_path / "b.md").write_text("beta notes, revised " * 20)
        changed = ingest_path(store, tmp_path, owner)
        assert changed["knowledge_epoch"] == 2
        assert changed["documents"] == 1 and changed["unchanged"] == 1
