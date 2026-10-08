from fastapi import FastAPI, HTTPException

from app.models.schemas import VectorSearchRequest, VectorSearchResponse
from app.services.ingestion_service import run_ingestion
from app.services.mongo import get_client
from app.services.structured_loader import load_structured_mongo
from app.services.vector_index import ensure_vector_index
from app.services.vector_search_service import search_vectors

app = FastAPI(
    title="Foxconn iPhone Vector API",
    version="1.1.0",
    description=(
        "Ingest Foxconn/iPhone synthetic knowledge (master, transactions, audit, "
        "and document chunks) and retrieve evidence with MongoDB Atlas Vector Search."
    ),
)


@app.get("/")
def root():
    return {"service": "Foxconn iPhone Vector API", "status": "running"}


@app.get("/health")
def health():
    client = get_client()
    try:
        ping = client.admin.command("ping")
        return {"status": "ok", "mongo": ping}
    finally:
        client.close()


@app.post("/api/v1/structured/ingest")
def ingest_structured():
    try:
        return load_structured_mongo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/vector/ingest")
def ingest_documents():
    try:
        return run_ingestion()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/vector/index")
def create_index():
    try:
        return ensure_vector_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/vector/search", response_model=VectorSearchResponse)
def vector_search(request: VectorSearchRequest):
    try:
        return search_vectors(
            query=request.query,
            top_k=request.top_k,
            bucket=request.bucket,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
