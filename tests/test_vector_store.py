
import pytest
from langchain_core.documents import Document

from modules.vector_store import (
    add_chunks,
    get_document_by_source,
    get_all_documents,
    delete_document,
    index_document,
    index_web_source,
    get_all_web_sources,
    delete_web_source,
    get_document_count,
)


class FakeVectorStore:
    """In-memory vector-store mock for automated tests."""

    def __init__(self):
        self.records = {}

    def add_documents(self, documents, ids):
        for document, chunk_id in zip(documents, ids):
            self.records[chunk_id] = {
                "document": document.page_content,
                "metadata": document.metadata.copy(),
            }

    def get(self, where=None, include=None):
        records = self.records

        if where:
            key, value = next(iter(where.items()))
            records = {
                chunk_id: record
                for chunk_id, record in records.items()
                if record["metadata"].get(key) == value
            }

        return {
            "ids": list(records.keys()),
            "metadatas": [
                record["metadata"].copy()
                for record in records.values()
            ],
        }

    def delete(self, ids):
        for chunk_id in ids:
            self.records.pop(chunk_id, None)


def make_chunk(source, file_id="hash123", source_type=None):
    metadata = {
        "source": source,
        "file_id": file_id,
        "file_type": "txt",
    }

    if source_type:
        metadata["source_type"] = source_type

    return Document(
        page_content="NEXUS AI test content",
        metadata=metadata,
    )


def test_add_chunks_success():
    store = FakeVectorStore()
    chunks = [make_chunk("sample.txt")]
    result = add_chunks(store, chunks, ["id1"])

    assert result == 1
    assert get_document_count(store) == 1


def test_add_chunks_rejects_empty_input():
    with pytest.raises(ValueError, match="no text chunks"):
        add_chunks(FakeVectorStore(), [], [])


def test_add_chunks_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="must match"):
        add_chunks(
            FakeVectorStore(),
            [make_chunk("sample.txt")],
            [],
        )


def test_document_indexing_adds_new_source():
    store = FakeVectorStore()

    result = index_document(
        store,
        [make_chunk("sample.txt")],
        ["id1"],
        "sample.txt",
        "hash123",
    )

    assert result == {"status": "added", "added": 1, "removed": 0}


def test_document_indexing_detects_duplicate():
    store = FakeVectorStore()
    chunk = make_chunk("sample.txt")

    index_document(store, [chunk], ["id1"], "sample.txt", "hash123")

    result = index_document(
        store,
        [chunk],
        ["id2"],
        "sample.txt",
        "hash123",
    )

    assert result["status"] == "duplicate"
    assert get_document_count(store) == 1


def test_document_indexing_replaces_old_version():
    store = FakeVectorStore()

    index_document(
        store,
        [make_chunk("sample.txt", "oldhash")],
        ["old1"],
        "sample.txt",
        "oldhash",
    )

    result = index_document(
        store,
        [make_chunk("sample.txt", "newhash")],
        ["new1"],
        "sample.txt",
        "newhash",
    )

    assert result == {"status": "replaced", "added": 1, "removed": 1}
    assert get_document_count(store) == 1
    assert "new1" in store.records
    assert "old1" not in store.records


def test_document_inventory_groups_chunks_by_source():
    store = FakeVectorStore()

    add_chunks(
        store,
        [
            make_chunk("sample.txt"),
            make_chunk("sample.txt"),
            make_chunk("report.pdf", "hash456"),
        ],
        ["id1", "id2", "id3"],
    )

    documents = get_all_documents(store)

    assert len(documents) == 2

    sample = next(item for item in documents if item["source"] == "sample.txt")
    assert sample["chunk_count"] == 2


def test_delete_document_removes_all_source_chunks():
    store = FakeVectorStore()

    add_chunks(
        store,
        [make_chunk("sample.txt"), make_chunk("sample.txt")],
        ["id1", "id2"],
    )

    removed = delete_document(store, "sample.txt")

    assert removed == 2
    assert get_document_count(store) == 0


def test_delete_missing_document_returns_zero():
    assert delete_document(FakeVectorStore(), "missing.txt") == 0


def test_web_indexing_does_not_replace_uploaded_document():
    store = FakeVectorStore()

    index_document(
        store,
        [make_chunk("https://example.com")],
        ["id1"],
        "https://example.com",
        "hash123",
    )

    with pytest.raises(ValueError, match="another source type"):
        index_web_source(
            store,
            [make_chunk("https://example.com", "webhash", "web_page")],
            ["id2"],
            "https://example.com",
            "webhash",
        )


def test_web_source_indexing_and_deletion():
    store = FakeVectorStore()

    web_chunk = make_chunk(
        "https://example.com",
        "webhash",
        "web_page",
    )

    result = index_web_source(
        store,
        [web_chunk],
        ["web1"],
        "https://example.com",
        "webhash",
    )

    assert result["status"] == "added"
    assert len(get_all_web_sources(store)) == 1

    removed = delete_web_source(store, "https://example.com")

    assert removed == 1
    assert get_document_count(store) == 0
