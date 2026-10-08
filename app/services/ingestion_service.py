import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pymongo import UpdateOne
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    MONGO_CHUNKS_COLLECTION,
    MONGO_DB_NAME,
)
from app.models.schemas import AuditLog
from app.services.chunking import chunk_document, parse_header_metadata
from app.services.dataset import resolve_dataset_root
from app.services.mongo import ensure_collections, get_client

_model = None
SUPPORTED = {".txt", ".pdf", ".docx"}
BUCKET_DIRS = (
    "product_specs",
    "internal_departments",
    "external_bodies",
    "unstructured",
)


def get_model(model_name: str = EMBEDDING_MODEL_NAME):
    global _model
    if _model is None or getattr(_model, "_name", None) != model_name:
        _model = SentenceTransformer(model_name)
        _model._name = model_name
    return _model


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return extract_txt(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".docx":
        return extract_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def bucket_for_path(dataset_root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(dataset_root).as_posix()
    except ValueError:
        relative = path.as_posix()
    top = relative.split("/", 1)[0]
    if top in BUCKET_DIRS:
        return top
    return "unstructured"


def iter_source_files(dataset_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for bucket in BUCKET_DIRS:
        root = dataset_root / bucket
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            if path.name.upper().startswith("PUBLIC_"):
                continue
            files.append((path, bucket))
    return files


def _lookup_document(db, document_id: str) -> dict:
    if not document_id:
        return {}
    found = db["documents"].find_one({"document_id": document_id}, {"_id": 0})
    return found or {}


def _chunk_records(
    metadata: dict,
    chunks: list[dict],
    embeddings,
    source_hash: str,
    relative_path: str,
) -> list[dict]:
    documents = []
    document_id = metadata["document_id"]
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_kind = chunk.get("chunk_kind") or "section"
        chunk_id = f"{document_id}::{chunk_kind}::{index + 1:04d}"
        documents.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": chunk["content"],
            "embedding": embedding.tolist(),
            "product": metadata.get("product"),
            "product_id": metadata.get("product_id"),
            "domain": metadata.get("domain"),
            "department": metadata.get("department"),
            "document_type": metadata.get("document_type"),
            "category": metadata.get("category") or metadata.get("topic"),
            "revision": metadata.get("revision"),
            "status": metadata.get("status") or "active",
            "file_name": metadata.get("file_name"),
            "relative_path": relative_path,
            "source_type": metadata.get("source_type"),
            "bucket": metadata.get("bucket"),
            "topic": metadata.get("topic"),
            "chunk_kind": chunk_kind,
            "parameter": chunk.get("parameter"),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "chunk_index": index,
            "chunk_count": len(chunks),
            "content_hash": source_hash,
        })
    return documents


def run_ingestion(
    strategy: str = "table_section",
    include_structured_sentences: bool = True,
) -> dict:
    dataset_root = resolve_dataset_root()
    started = datetime.now(timezone.utc)
    run_id = f"unstructured-{uuid4().hex[:12]}"
    client = get_client()
    processed = 0
    skipped_duplicates = 0
    failed = 0
    chunks_inserted = 0
    errors: list[dict] = []
    seen_hashes: set[str] = set()

    try:
        db = client[MONGO_DB_NAME]
        ensure_collections(db)
        collection = db[MONGO_CHUNKS_COLLECTION]
        model = get_model()
        files = iter_source_files(dataset_root)

        for path, bucket in files:
            try:
                raw_text = extract_text(path)
                if not raw_text.strip():
                    failed += 1
                    errors.append({"file": str(path), "error": "Empty extracted text"})
                    continue
                source_hash = content_hash(raw_text)
                if source_hash in seen_hashes:
                    skipped_duplicates += 1
                    continue
                seen_hashes.add(source_hash)

                relative_path = path.relative_to(dataset_root).as_posix()
                metadata = parse_header_metadata(raw_text, path, bucket)
                catalog = _lookup_document(db, metadata.get("document_id", ""))
                if catalog:
                    metadata.update({
                        key: catalog.get(key) or metadata.get(key)
                        for key in (
                            "product_id",
                            "product",
                            "department",
                            "domain",
                            "revision",
                            "status",
                            "topic",
                            "document_type",
                            "bucket",
                        )
                    })
                    metadata["product"] = catalog.get("product_name") or metadata.get("product")
                metadata["bucket"] = metadata.get("bucket") or bucket
                metadata.setdefault("status", "active")
                metadata["relative_path"] = relative_path

                chunks = chunk_document(raw_text, metadata, strategy=strategy)
                if not chunks:
                    failed += 1
                    errors.append({"file": str(path), "error": "No chunks generated"})
                    continue

                embeddings = model.encode(
                    [item["content"] for item in chunks],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                desired = _chunk_records(
                    metadata, chunks, embeddings, source_hash, relative_path
                )
                collection.bulk_write(
                    [
                        UpdateOne({"chunk_id": doc["chunk_id"]}, {"$set": doc}, upsert=True)
                        for doc in desired
                    ],
                    ordered=True,
                )
                collection.delete_many({
                    "document_id": metadata["document_id"],
                    "chunk_id": {"$nin": [doc["chunk_id"] for doc in desired]},
                    "source_type": {"$ne": "structured_sentence"},
                })
                processed += 1
                chunks_inserted += len(desired)
            except Exception as exc:
                failed += 1
                errors.append({"file": str(path), "error": str(exc)})

        structured_chunks = 0
        if include_structured_sentences:
            structured_chunks = _ingest_structured_sentences(db, collection, model)

        finished = datetime.now(timezone.utc)
        status = "success" if failed == 0 else "completed_with_errors"
        audit = AuditLog(
            run_id=run_id,
            job_type="unstructured_ingest",
            started_at=started,
            finished_at=finished,
            status=status,
            counts={
                "source_documents_found": len(files),
                "documents_processed": processed,
                "duplicates_skipped": skipped_duplicates,
                "failed": failed,
                "chunks_inserted": chunks_inserted,
                "structured_sentence_chunks": structured_chunks,
            },
            errors=errors,
        )
        db["audit_log"].update_one(
            {"run_id": run_id},
            {"$set": audit.model_dump()},
            upsert=True,
        )
        return {
            "status": status,
            "run_id": run_id,
            "dataset_root": str(dataset_root),
            "source_documents_found": len(files),
            "documents_processed": processed,
            "duplicates_skipped": skipped_duplicates,
            "failed": failed,
            "chunks_inserted": chunks_inserted,
            "structured_sentence_chunks": structured_chunks,
            "errors": errors,
        }
    finally:
        client.close()


def _ingest_structured_sentences(db, collection, model) -> int:
    rows = list(db["transactions"].find({}, {"_id": 0}))
    if not rows:
        return 0
    contents = [row.get("canonical_sentence") or "" for row in rows]
    embeddings = model.encode(
        contents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    desired = []
    for index, (row, embedding) in enumerate(zip(rows, embeddings)):
        document_id = row.get("source_document_id") or row["transaction_id"]
        chunk_id = f"{row['transaction_id']}::spec_row::0001"
        catalog = _lookup_document(db, row.get("source_document_id") or "")
        desired.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": row.get("canonical_sentence"),
            "embedding": embedding.tolist(),
            "product": catalog.get("product_name"),
            "product_id": row.get("product_id"),
            "domain": catalog.get("domain") or "structured_data",
            "department": catalog.get("department"),
            "document_type": catalog.get("document_type"),
            "revision": catalog.get("revision"),
            "status": catalog.get("status") or "active",
            "source_type": "structured_sentence",
            "bucket": "structured_data",
            "topic": row.get("attribute"),
            "chunk_kind": "spec_row",
            "parameter": row.get("attribute"),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "chunk_index": 0,
            "chunk_count": 1,
            "transaction_id": row.get("transaction_id"),
        })
    collection.bulk_write(
        [
            UpdateOne({"chunk_id": doc["chunk_id"]}, {"$set": doc}, upsert=True)
            for doc in desired
        ]
    )
    return len(desired)
