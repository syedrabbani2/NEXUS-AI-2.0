from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings():
    """Load a small sentence-transformer model locally on the computer."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
