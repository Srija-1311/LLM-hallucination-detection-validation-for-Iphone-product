from fastapi import FastAPI, HTTPException

from app.models.schemas import (
    VectorSearchRequest,
    VectorSearchResponse,
)
from app.services.ingestion_service import run_ingestion
from app.services.vector_search_service import search_vectors


app = FastAPI(
    title="Foxconn iPhone Vector API",
    version="1.0.0",
    description=(
        "APIs for ingesting unstructured synthetic Foxconn/iPhone documents "
        "and retrieving relevant evidence using MongoDB Atlas Vector Search."
    )
)


@app.get("/")
def root():
    return {
        "service": "Foxconn iPhone Vector API",
        "status": "running"
    }


@app.post("/api/v1/vector/ingest")
def ingest_documents():
    try:
        return run_ingestion()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.post(
    "/api/v1/vector/search",
    response_model=VectorSearchResponse
)
def vector_search(request: VectorSearchRequest):
    try:
        return search_vectors(
            query=request.query,
            top_k=request.top_k
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
