import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection("data_embeddings")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def store_vectors_in_chromadb(text_chunks, source_name):
    """Stores extracted text chunks as vector embeddings in ChromaDB."""
    for i, chunk in enumerate(text_chunks):
        collection.add(
            ids=[f"{source_name}_chunk_{i}"],
            documents=[chunk],
            metadatas=[{"source": source_name}],
        )

def retrieve_relevant_chunks(query):
    """Finds relevant text chunks from ChromaDB based on query similarity."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if not results["documents"]:
        return []
    
    references = []
    for doc, metadata in zip(results["documents"], results["metadatas"]):
        source_name = metadata[0].get("source", "Unknown Source") if metadata and metadata[0] else "Unknown Source"
        references.append((source_name, doc))
    
    return references
