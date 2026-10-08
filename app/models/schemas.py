from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    bucket: Optional[str] = Field(
        default=None,
        description="Optional Atlas pre-filter: product_specs, internal_departments, external_bodies, unstructured, structured_data",
    )


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


class Product(BaseModel):
    product_id: str
    product_name: str
    product_family: Optional[str] = None
    model_year: Optional[int] = None
    synthetic_dataset_flag: bool = True
    source_type: str = "synthetic"


class KbDocument(BaseModel):
    document_id: str
    product_id: str
    product_name: Optional[str] = None
    bucket: str
    domain: Optional[str] = None
    organization: Optional[str] = None
    department: Optional[str] = None
    document_type: Optional[str] = None
    revision: Optional[str] = None
    status: str = "active"
    topic: Optional[str] = None
    file_path: Optional[str] = None
    source_type: str = "synthetic"
    confidentiality: Optional[str] = None
    date: Optional[str] = None


class Transaction(BaseModel):
    transaction_id: str
    product_id: str
    source_table: str
    attribute: str
    value: Optional[str] = None
    unit: Optional[str] = None
    source_document_id: Optional[str] = None
    evidence_status: Optional[str] = None
    canonical_sentence: str
    raw: dict[str, Any] = Field(default_factory=dict)


class AuditLog(BaseModel):
    run_id: str
    job_type: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: str = "running"
    counts: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
