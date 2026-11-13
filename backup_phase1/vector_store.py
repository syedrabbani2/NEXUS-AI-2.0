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
    if not chunks:
        raise ValueError("There are no text chunks to add.")
    vector_store.add_documents(documents=chunks, ids=ids)
    return len(chunks)
