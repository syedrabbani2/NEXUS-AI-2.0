from collections import defaultdict
from config import CHROMA_DIR
from modules.embeddings import create_embeddings
from modules.vector_store import get_vector_store

vector_store = get_vector_store(
    create_embeddings(),
    str(CHROMA_DIR)
)

result = vector_store.get(include=["metadatas"])

documents = defaultdict(lambda: {
    "chunks": 0,
    "file_id": None,
    "type": None
})

for metadata in result["metadatas"]:
    name = metadata.get("source", "Unknown")

    documents[name]["chunks"] += 1
    documents[name]["file_id"] = metadata.get("file_id")
    documents[name]["type"] = metadata.get("file_type")

print("\nTOTAL INDEXED CHUNKS:", len(result["ids"]))
print("UNIQUE DOCUMENTS:", len(documents))
print("\nDOCUMENT INVENTORY:\n")

for name, details in sorted(documents.items()):
    print("Name:", name)
    print("Chunks:", details["chunks"])
    print("Type:", details["type"])
    print("File ID:", details["file_id"])
    print("-" * 50)
