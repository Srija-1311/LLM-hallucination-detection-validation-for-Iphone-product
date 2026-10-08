from typing import Optional

from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    MONGO_CHUNKS_COLLECTION,
    MONGO_DB_NAME,
    VECTOR_INDEX_NAME,
)
from app.services.mongo import get_client

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def search_vectors(query: str, top_k: int = 5, bucket: Optional[str] = None) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    query_embedding = get_model().encode(query, normalize_embeddings=True).tolist()
    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Unexpected embedding dimension: {len(query_embedding)}. "
            f"Expected {EMBEDDING_DIMENSION}."
        )

    search_stage = {
        "index": VECTOR_INDEX_NAME,
        "path": "embedding",
        "queryVector": query_embedding,
        "numCandidates": max(top_k * 10, 50),
        "limit": top_k,
    }
    if bucket:
        search_stage["filter"] = {"bucket": {"$eq": bucket}}

    pipeline = [
        {"$vectorSearch": search_stage},
        {
            "$project": {
                "_id": 0,
                "document_id": 1,
                "chunk_id": 1,
                "content": 1,
                "product": 1,
                "product_id": 1,
                "domain": 1,
                "department": 1,
                "document_type": 1,
                "category": 1,
                "revision": 1,
                "status": 1,
                "file_name": 1,
                "relative_path": 1,
                "source_type": 1,
                "bucket": 1,
                "topic": 1,
                "chunk_kind": 1,
                "parameter": 1,
                "embedding_model": 1,
                "embedding_dimension": 1,
                "chunk_index": 1,
                "chunk_count": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    client = get_client()
    try:
        results = list(client[MONGO_DB_NAME][MONGO_CHUNKS_COLLECTION].aggregate(pipeline))
        formatted = []
        for result in results:
            formatted.append({
                "document_id": result.get("document_id", ""),
                "chunk_id": result.get("chunk_id", ""),
                "score": float(result.get("score", 0.0)),
                "content": result.get("content", ""),
                "metadata": {
                    "product": result.get("product"),
                    "product_id": result.get("product_id"),
                    "domain": result.get("domain"),
                    "department": result.get("department"),
                    "document_type": result.get("document_type"),
                    "category": result.get("category"),
                    "revision": result.get("revision"),
                    "status": result.get("status"),
                    "file_name": result.get("file_name"),
                    "relative_path": result.get("relative_path"),
                    "source_type": result.get("source_type"),
                    "bucket": result.get("bucket"),
                    "topic": result.get("topic"),
                    "chunk_kind": result.get("chunk_kind"),
                    "parameter": result.get("parameter"),
                    "embedding_model": result.get("embedding_model"),
                    "embedding_dimension": result.get("embedding_dimension"),
                    "chunk_index": result.get("chunk_index"),
                    "chunk_count": result.get("chunk_count"),
                },
            })
        return {"status": "success", "query": query, "results": formatted}
    finally:
        client.close()
