
import math
import re
from collections import Counter

from langchain_core.documents import Document


# BM25 tuning parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal Rank Fusion parameter
RRF_K = 60


def tokenize(text):
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def bm25_scores(query, documents):
    """
    Calculate BM25-style relevance scores for a document collection.

    Returns a list of (document_index, score) tuples.
    """

    query_terms = tokenize(query)

    if not query_terms or not documents:
        return []

    tokenized_docs = [
        tokenize(doc.page_content)
        for doc in documents
    ]

    total_docs = len(tokenized_docs)

    if total_docs == 0:
        return []

    doc_lengths = [
        len(tokens)
        for tokens in tokenized_docs
    ]

    average_length = (
        sum(doc_lengths) / total_docs
        if total_docs else 0
    )

    document_frequency = Counter()

    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    query_frequency = Counter(query_terms)

    scores = []

    for index, tokens in enumerate(tokenized_docs):

        frequencies = Counter(tokens)
        score = 0.0

        for term, query_count in query_frequency.items():

            term_frequency = frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            doc_frequency = document_frequency[term]

            inverse_document_frequency = math.log(
                1 + (
                    total_docs - doc_frequency + 0.5
                ) / (
                    doc_frequency + 0.5
                )
            )

            length_normalization = (
                term_frequency
                + BM25_K1 * (
                    1 - BM25_B
                    + BM25_B * (
                        doc_lengths[index] / average_length
                        if average_length else 0
                    )
                )
            )

            term_score = (
                inverse_document_frequency
                * term_frequency
                * (BM25_K1 + 1)
                / length_normalization
            )

            score += query_count * term_score

        if score > 0:
            scores.append((index, score))

    return sorted(
        scores,
        key=lambda item: item[1],
        reverse=True
    )


def document_key(document):
    """Create a stable key for identifying duplicate chunks."""

    metadata = document.metadata or {}

    return (
        metadata.get("source"),
        metadata.get("file_id"),
        metadata.get("chunk_index"),
        metadata.get("page_number"),
        metadata.get("row_number"),
        metadata.get("sheet_name"),
        document.page_content
    )


def reciprocal_rank_fusion(
    ranked_lists,
    top_k=4,
    rrf_k=RRF_K
):
    """
    Combine multiple ranked document lists using RRF.
    """

    combined_scores = {}
    documents_by_key = {}

    for ranked_documents in ranked_lists:

        for rank, document in enumerate(
            ranked_documents,
            start=1
        ):

            key = document_key(document)

            combined_scores[key] = (
                combined_scores.get(key, 0.0)
                + 1.0 / (rrf_k + rank)
            )

            if key not in documents_by_key:
                documents_by_key[key] = document

    ordered_keys = sorted(
        combined_scores,
        key=lambda key: combined_scores[key],
        reverse=True
    )

    return [
        documents_by_key[key]
        for key in ordered_keys[:top_k]
    ]


def hybrid_search(
    query,
    vector_store,
    top_k=4,
    fetch_k=20
):
    """
    Retrieve documents using semantic and BM25-style keyword search,
    then combine rankings using Reciprocal Rank Fusion.

    Existing Chroma documents and metadata are preserved.
    """

    if not query or not query.strip():
        return []

    if top_k < 1 or fetch_k < 1:
        raise ValueError("top_k and fetch_k must be positive.")

    # 1. Semantic retrieval from ChromaDB
    semantic_results = vector_store.similarity_search(
        query,
        k=fetch_k
    )

    # 2. Load indexed document text and metadata for keyword retrieval
    collection = vector_store.get(
        include=["documents", "metadatas"]
    )

    raw_documents = collection.get("documents") or []
    raw_metadatas = collection.get("metadatas") or []

    keyword_documents = []

    for content, metadata in zip(
        raw_documents,
        raw_metadatas
    ):

        if content is None:
            continue

        keyword_documents.append(
            Document(
                page_content=content,
                metadata=metadata or {}
            )
        )

    # 3. Calculate BM25-style keyword ranking
    scored_documents = bm25_scores(
        query,
        keyword_documents
    )

    keyword_results = [
        keyword_documents[index]
        for index, score in scored_documents[:fetch_k]
    ]

    # 4. Merge semantic and keyword rankings
    return reciprocal_rank_fusion(
        ranked_lists=[
            semantic_results,
            keyword_results
        ],
        top_k=top_k
    )
