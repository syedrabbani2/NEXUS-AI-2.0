
from langchain_chroma import Chroma

COLLECTION_NAME = "nexus_enterprise_knowledge"


def get_vector_store(embeddings, persist_directory: str):
    """Open or create the persistent local Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def add_chunks(vector_store, chunks, ids):
    """Add document chunks to the vector database."""
    if not chunks:
        raise ValueError("There are no text chunks to add.")

    if len(chunks) != len(ids):
        raise ValueError("The number of chunks and IDs must match.")

    vector_store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def get_document_by_source(vector_store, source):
    """Find all indexed chunks belonging to a source."""

    result = vector_store.get(
        where={"source": source},
        include=["metadatas"]
    )

    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])

    if not ids:
        return None

    first_metadata = metadatas[0] if metadatas else {}

    return {
        "source": source,
        "file_id": first_metadata.get("file_id"),
        "file_type": first_metadata.get("file_type"),
        "source_type": first_metadata.get("source_type", "uploaded_document"),
        "title": first_metadata.get("title"),
        "url": first_metadata.get("url"),
        "retrieved_at": first_metadata.get("retrieved_at"),
        "chunk_count": len(ids),
        "ids": ids,
        "metadatas": metadatas
    }


def get_all_documents(vector_store):
    """Build an inventory of indexed sources."""

    result = vector_store.get(include=["metadatas"])
    documents = {}

    for chunk_id, metadata in zip(
        result.get("ids", []),
        result.get("metadatas", [])
    ):
        source = metadata.get("source", "Unknown document")

        if source not in documents:
            documents[source] = {
                "source": source,
                "file_id": metadata.get("file_id"),
                "file_type": metadata.get("file_type"),
                "source_type": metadata.get(
                    "source_type", "uploaded_document"
                ),
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "retrieved_at": metadata.get("retrieved_at"),
                "chunk_count": 0,
                "ids": []
            }

        document = documents[source]
        document["chunk_count"] += 1
        document["ids"].append(chunk_id)

        for field in (
            "file_type",
            "file_id",
            "source_type",
            "title",
            "url",
            "retrieved_at"
        ):
            if not document.get(field) and metadata.get(field):
                document[field] = metadata[field]

    for document in documents.values():
        if not document["file_type"]:
            document["file_type"] = (
                document["source"].rsplit(".", 1)[-1].lower()
                if "." in document["source"]
                else "unknown"
            )

    return sorted(
        documents.values(),
        key=lambda item: item["source"].lower()
    )


def delete_document(vector_store, source):
    """Delete every indexed chunk belonging to a source."""

    document = get_document_by_source(vector_store, source)

    if not document:
        return 0

    vector_store.delete(ids=document["ids"])
    return document["chunk_count"]


def index_document(vector_store, chunks, ids, source, file_id):
    """Index uploaded documents without affecting web sources."""

    if not chunks:
        raise ValueError("There are no document chunks to index.")

    existing = get_document_by_source(vector_store, source)

    if existing and existing["file_id"] == file_id:
        return {
            "status": "duplicate",
            "added": 0,
            "removed": 0
        }

    old_ids = existing["ids"] if existing else []

    added = add_chunks(vector_store, chunks, ids)

    removed = 0

    if old_ids:
        vector_store.delete(ids=old_ids)
        removed = len(old_ids)

    return {
        "status": "replaced" if existing else "added",
        "added": added,
        "removed": removed
    }


def index_web_source(vector_store, chunks, ids, url, file_id):
    """
    Index or refresh a web page.

    Only chunks marked as web_page are eligible for replacement.
    """

    if not chunks:
        raise ValueError("There are no web chunks to index.")

    existing = get_document_by_source(vector_store, url)

    if existing and existing["source_type"] != "web_page":
        raise ValueError(
            "This source belongs to another source type. "
            "Web indexing was stopped."
        )

    if existing and existing["file_id"] == file_id:
        return {
            "status": "unchanged",
            "added": 0,
            "removed": 0
        }

    old_ids = existing["ids"] if existing else []

    added = add_chunks(vector_store, chunks, ids)

    removed = 0

    if old_ids:
        vector_store.delete(ids=old_ids)
        removed = len(old_ids)

    return {
        "status": "refreshed" if existing else "added",
        "added": added,
        "removed": removed
    }


def get_all_web_sources(vector_store):
    """Return only indexed web pages."""

    return [
        document
        for document in get_all_documents(vector_store)
        if document["source_type"] == "web_page"
    ]


def delete_web_source(vector_store, url):
    """Delete a web page without affecting uploaded documents."""

    document = get_document_by_source(vector_store, url)

    if not document:
        return 0

    if document["source_type"] != "web_page":
        raise ValueError("The selected source is not a web page.")

    vector_store.delete(ids=document["ids"])
    return document["chunk_count"]


def get_document_count(vector_store):
    """Return the total number of indexed chunks."""

    result = vector_store.get()
    return len(result.get("ids", []))
