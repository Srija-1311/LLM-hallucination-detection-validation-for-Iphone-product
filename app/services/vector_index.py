from pymongo.operations import SearchIndexModel

from app.config import (
    EMBEDDING_DIMENSION,
    MONGO_CHUNKS_COLLECTION,
    MONGO_DB_NAME,
    VECTOR_INDEX_NAME,
)
from app.services.mongo import get_client


def build_vector_index_model() -> SearchIndexModel:
    return SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBEDDING_DIMENSION,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "bucket"},
                {"type": "filter", "path": "status"},
            ]
        },
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )


def ensure_vector_index() -> dict:
    client = get_client()
    try:
        collection = client[MONGO_DB_NAME][MONGO_CHUNKS_COLLECTION]
        existing = []
        try:
            existing = list(collection.list_search_indexes())
        except Exception as exc:
            existing_error = str(exc)
        else:
            existing_error = None

        names = {item.get("name") for item in existing}
        dropped = False
        if VECTOR_INDEX_NAME in names:
            collection.drop_search_index(VECTOR_INDEX_NAME)
            dropped = True

        result = collection.create_search_index(model=build_vector_index_model())
        return {
            "status": "requested",
            "index_name": result,
            "replaced_existing": dropped,
            "list_error": existing_error,
        }
    finally:
        client.close()
