
import pytest
from langchain_core.documents import Document

from modules.hybrid_retrieval import (
    tokenize,
    bm25_scores,
    document_key,
    reciprocal_rank_fusion,
    hybrid_search,
)


def make_document(content, source="sample.txt", index=0):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "file_id": "hash123",
            "chunk_index": index,
        },
    )


class FakeVectorStore:
    def __init__(self, documents, semantic_results=None):
        self.documents = documents
        self.semantic_results = semantic_results or []

    def similarity_search(self, query, k=4):
        return self.semantic_results[:k]

    def get(self, include=None):
        return {
            "documents": [doc.page_content for doc in self.documents],
            "metadatas": [doc.metadata for doc in self.documents],
        }


def test_tokenize_lowercases_and_extracts_tokens():
    assert tokenize("NEXUS AI Version 2.0!") == [
        "nexus", "ai", "version", "2", "0"
    ]


def test_tokenize_handles_empty_input():
    assert tokenize("") == []


def test_bm25_returns_relevant_documents():
    documents = [
        make_document("Python machine learning AI"),
        make_document("School transport management"),
        make_document("Python AI development"),
    ]

    scores = bm25_scores("Python AI", documents)

    assert len(scores) == 2
    assert scores[0][1] >= scores[1][1]


def test_bm25_returns_empty_for_no_matches():
    documents = [
        make_document("School transport"),
        make_document("Fee management"),
    ]

    assert bm25_scores("quantum physics", documents) == []


def test_bm25_handles_empty_query():
    documents = [make_document("NEXUS AI")]

    assert bm25_scores("", documents) == []


def test_document_key_identifies_same_chunk():
    first = make_document("NEXUS AI", index=1)
    second = make_document("NEXUS AI", index=1)

    assert document_key(first) == document_key(second)


def test_rrf_removes_duplicate_documents():
    first = make_document("NEXUS AI", index=1)
    second = make_document("NEXUS AI", index=1)

    result = reciprocal_rank_fusion(
        [[first], [second]],
        top_k=4,
    )

    assert len(result) == 1


def test_rrf_respects_top_k():
    documents = [
        make_document("Python", index=1),
        make_document("Java", index=2),
        make_document("SQL", index=3),
    ]

    result = reciprocal_rank_fusion(
        [documents],
        top_k=2,
    )

    assert len(result) == 2


def test_hybrid_search_returns_ranked_results():
    documents = [
        make_document("Python AI development", index=1),
        make_document("School transport management", index=2),
    ]

    store = FakeVectorStore(
        documents,
        semantic_results=[documents[1]],
    )

    results = hybrid_search(
        "Python AI",
        store,
        top_k=2,
        fetch_k=10,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_hybrid_search_handles_empty_query():
    store = FakeVectorStore([])

    assert hybrid_search("", store) == []


def test_hybrid_search_rejects_invalid_limits():
    store = FakeVectorStore([])

    with pytest.raises(ValueError, match="must be positive"):
        hybrid_search("Python", store, top_k=0)
