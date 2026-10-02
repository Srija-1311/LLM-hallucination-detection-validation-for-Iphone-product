from typing import Optional

from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    content: str
    metadata: dict


class VectorSearchResponse(BaseModel):
    status: str
    query: str
    results: list[SearchResult]
