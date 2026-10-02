import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# Keep this path aligned with the existing working ingestion script.
DATASET_ROOT = Path(
    "C:/Users/srija/OneDrive/ドキュメント/"
    "foxconn synthetic data/foxconn synthetic data"
)
UNSTRUCTURED_ROOT = DATASET_ROOT / "unstructured"

MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "foxconn_iphone_kb")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "document_chunks")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalized_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(
        normalized_for_hash(text).encode("utf-8")
    ).hexdigest()


def chunk_text(text: str) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return extract_txt(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".docx":
        return extract_docx(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def parse_header_metadata(text: str, path: Path) -> Dict[str, str]:
    fields = [
        "Document ID",
        "Product",
        "Domain",
        "Department",
        "Document Type",
        "Category",
        "Revision",
        "Status",
    ]

    metadata = {}

    for field in fields:
        match = re.search(
            rf"(?im)^{re.escape(field)}:\s*(.+?)\s*$",
            text
        )
        if match:
            metadata[field.lower().replace(" ", "_")] = match.group(1).strip()

    relative_path = str(path.relative_to(UNSTRUCTURED_ROOT))

    metadata.setdefault("file_name", path.name)
    metadata.setdefault("relative_path", relative_path)
    metadata.setdefault("source_type", "unstructured")

    # Fallback if a document does not contain a Document ID header.
    metadata.setdefault(
        "document_id",
        path.relative_to(UNSTRUCTURED_ROOT).with_suffix("").as_posix()
    )

    return metadata


def get_document_files() -> List[Path]:
    supported = {".txt", ".pdf", ".docx"}

    return sorted(
        p for p in UNSTRUCTURED_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    )


def run_ingestion() -> Dict:
    if not MONGO_URI:
        raise RuntimeError(
            "MONGODB_URI (or MONGO_URI) is missing from the .env file."
        )

    if not UNSTRUCTURED_ROOT.exists():
        raise FileNotFoundError(
            f"Unstructured dataset folder not found: {UNSTRUCTURED_ROOT}"
        )

    files = get_document_files()

    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]

    # Safe/idempotent indexes.
    collection.create_index(
        [("document_id", 1)],
        name="document_id_index"
    )

    collection.create_index(
        [("chunk_id", 1)],
        name="unique_chunk_id",
        unique=True
    )

    model = get_model()

    processed = 0
    skipped_duplicates = 0
    failed = 0
    chunks_inserted = 0
    errors = []

    # Prevent exact duplicate source documents within this run.
    seen_content_hashes = set()

    try:
        for path in files:
            try:
                raw_text = extract_text(path)
                text = clean_text(raw_text)

                if not text:
                    failed += 1
                    errors.append({
                        "file": str(path),
                        "error": "Empty extracted text"
                    })
                    continue

                source_hash = content_hash(text)

                if source_hash in seen_content_hashes:
                    skipped_duplicates += 1
                    continue

                seen_content_hashes.add(source_hash)

                metadata = parse_header_metadata(text, path)
                document_id = metadata["document_id"]

                chunks = chunk_text(text)

                if not chunks:
                    failed += 1
                    errors.append({
                        "file": str(path),
                        "error": "No chunks generated"
                    })
                    continue

                embeddings = model.encode(
                    chunks,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                # Build the complete desired state for this document first.
                # We only modify MongoDB after extraction, chunking, and
                # embedding have all succeeded.
                desired_documents = []

                for index, (chunk, embedding) in enumerate(
                    zip(chunks, embeddings)
                ):
                    chunk_id = f"{document_id}::chunk-{index + 1:04d}"

                    desired_documents.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "content": chunk,
                        "embedding": embedding.tolist(),

                        "product": metadata.get("product"),
                        "domain": metadata.get("domain"),
                        "department": metadata.get("department"),
                        "document_type": metadata.get("document_type"),
                        "category": metadata.get("category"),
                        "revision": metadata.get("revision"),
                        "status": metadata.get("status"),

                        "file_name": metadata.get("file_name"),
                        "relative_path": metadata.get("relative_path"),
                        "source_type": metadata.get("source_type"),

                        "embedding_model": EMBEDDING_MODEL_NAME,
                        "embedding_dimension": EMBEDDING_DIMENSION,

                        "chunk_index": index,
                        "chunk_count": len(chunks),
                        "content_hash": source_hash
                    })

                # Safe reconciliation:
                # 1. Upsert every desired chunk.
                # 2. Remove only stale chunks belonging to this document.
                # If embedding generation failed, we never reach this point,
                # so the existing MongoDB data remains untouched.
                from pymongo import UpdateOne

                operations = [
                    UpdateOne(
                        {"chunk_id": doc["chunk_id"]},
                        {"$set": doc},
                        upsert=True
                    )
                    for doc in desired_documents
                ]

                if operations:
                    collection.bulk_write(operations, ordered=True)

                desired_chunk_ids = [
                    doc["chunk_id"] for doc in desired_documents
                ]

                collection.delete_many({
                    "document_id": document_id,
                    "chunk_id": {"$nin": desired_chunk_ids}
                })

                processed += 1
                chunks_inserted += len(desired_documents)

            except Exception as exc:
                failed += 1
                errors.append({
                    "file": str(path),
                    "error": str(exc)
                })

        return {
            "status": "success" if failed == 0 else "completed_with_errors",
            "source_documents_found": len(files),
            "documents_processed": processed,
            "duplicates_skipped": skipped_duplicates,
            "failed": failed,
            "chunks_inserted": chunks_inserted,
            "errors": errors
        }

    finally:
        client.close()
