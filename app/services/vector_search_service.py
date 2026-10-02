import os
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "foxconn_iphone_kb")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "document_chunks")

VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def search_vectors(query: str, top_k: int = 5) -> Dict:
    if not MONGO_URI:
        raise RuntimeError(
            "MONGODB_URI (or MONGO_URI) is missing from the .env file."
        )

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    model = get_model()

    # Generate the same 384-dimensional embedding used during ingestion.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Unexpected embedding dimension: {len(query_embedding)}. "
            f"Expected {EMBEDDING_DIMENSION}."
        )

    client = MongoClient(MONGO_URI)

    try:
        collection = client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(top_k * 10, 50),
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "document_id": 1,
                    "chunk_id": 1,
                    "content": 1,
                    "product": 1,
                    "domain": 1,
                    "department": 1,
                    "document_type": 1,
                    "category": 1,
                    "revision": 1,
                    "status": 1,
                    "file_name": 1,
                    "relative_path": 1,
                    "source_type": 1,
                    "embedding_model": 1,
                    "embedding_dimension": 1,
                    "chunk_index": 1,
                    "chunk_count": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        formatted_results = []

        for result in results:
            metadata = {
                "product": result.get("product"),
                "domain": result.get("domain"),
                "department": result.get("department"),
                "document_type": result.get("document_type"),
                "category": result.get("category"),
                "revision": result.get("revision"),
                "status": result.get("status"),
                "file_name": result.get("file_name"),
                "relative_path": result.get("relative_path"),
                "source_type": result.get("source_type"),
                "embedding_model": result.get("embedding_model"),
                "embedding_dimension": result.get("embedding_dimension"),
                "chunk_index": result.get("chunk_index"),
                "chunk_count": result.get("chunk_count")
            }

            formatted_results.append({
                "document_id": result.get("document_id", ""),
                "chunk_id": result.get("chunk_id", ""),
                "score": float(result.get("score", 0.0)),
                "content": result.get("content", ""),
                "metadata": metadata
            })

        return {
            "status": "success",
            "query": query,
            "results": formatted_results
        }

    finally:
        client.close()
