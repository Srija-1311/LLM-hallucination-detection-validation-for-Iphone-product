import os
import re
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "foxconn_iphone_kb")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "document_chunks")

if not MONGO_URI:
    raise ValueError(
        "MONGODB_URI is not set in the .env file."
    )


# ============================================================
# 2. DATASET PATH
# ============================================================

DATASET_ROOT = Path(
    "C:/Users/srija/OneDrive/ドキュメント/"
    "foxconn synthetic data/foxconn synthetic data"
)

UNSTRUCTURED_ROOT = DATASET_ROOT / "unstructured"


# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# 4. TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_path):
    reader = PdfReader(str(file_path))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_docx_text(file_path):
    document = Document(str(file_path))
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def extract_txt_text(file_path):
    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def extract_text(file_path):
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    if extension == ".docx":
        return extract_docx_text(file_path)

    if extension == ".txt":
        return extract_txt_text(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


# ============================================================
# 5. TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.replace("\x00", " ")

    # Normalize line endings and whitespace without destroying
    # the actual content.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalized_for_hash(text):
    """
    Used only for duplicate detection.
    It does not replace the actual text stored in MongoDB.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha256_text(text):
    return hashlib.sha256(
        normalized_for_hash(text).encode("utf-8")
    ).hexdigest()


# ============================================================
# 6. CHUNKING
# ============================================================

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# 7. METADATA
# ============================================================

def parse_header_metadata(text):
    """
    The synthetic TXT documents contain metadata such as:

        Document ID:
        Product:
        Domain:
        Department:
        Document Type:
        Category:

    Read these values from the document itself rather than
    assuming that folder names represent the metadata.
    """

    patterns = {
        "document_id": r"(?im)^\s*Document ID\s*:\s*(.+?)\s*$",
        "product": r"(?im)^\s*Product\s*:\s*(.+?)\s*$",
        "domain": r"(?im)^\s*Domain\s*:\s*(.+?)\s*$",
        "department": r"(?im)^\s*Department\s*:\s*(.+?)\s*$",
        "document_type": r"(?im)^\s*Document Type\s*:\s*(.+?)\s*$",
        "category": r"(?im)^\s*Category\s*:\s*(.+?)\s*$",
        "revision": r"(?im)^\s*Revision\s*:\s*(.+?)\s*$",
        "status": r"(?im)^\s*Status\s*:\s*(.+?)\s*$",
    }

    metadata = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        metadata[key] = match.group(1).strip() if match else None

    return metadata


def get_metadata(file_path, text):
    relative_path = file_path.relative_to(UNSTRUCTURED_ROOT)
    parts = relative_path.parts

    header = parse_header_metadata(text)

    # Use document-header metadata when available.
    # Fall back to the folder hierarchy for older/non-header files.
    document_type = header["document_type"]
    category = header["category"]
    domain = header["domain"]
    department = header["department"]

    if document_type is None and len(parts) >= 1:
        document_type = parts[0]

    if category is None and len(parts) >= 2:
        category = parts[1]

    return {
        "document_id_from_header": header["document_id"],
        "product": header["product"],
        "domain": domain,
        "department": department,
        "document_type": document_type,
        "category": category,
        "revision": header["revision"],
        "status": header["status"],
        "file_name": file_path.name,
        "relative_path": str(relative_path).replace("\\", "/"),
    }


# ============================================================
# 8. DOCUMENT ID
# ============================================================

def create_document_id(file_path, metadata):
    """
    Prefer the explicit Document ID inside the document.
    Fall back to the relative path if the document has no ID.
    """

    if metadata.get("document_id_from_header"):
        return metadata["document_id_from_header"]

    relative_path = file_path.relative_to(UNSTRUCTURED_ROOT)
    document_id = str(relative_path).replace("\\", "/")

    document_id = re.sub(
        r"[^a-zA-Z0-9/_\-.]",
        "_",
        document_id
    )

    return document_id


# ============================================================
# 9. MAIN INGESTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FOXCONN iPHONE UNSTRUCTURED DOCUMENT INGESTION")
    print("IDEMPOTENT + DUPLICATE-AWARE VERSION")
    print("=" * 70)

    if not UNSTRUCTURED_ROOT.exists():
        raise FileNotFoundError(
            f"Unstructured dataset not found:\n{UNSTRUCTURED_ROOT}"
        )

    print(f"\nScanning:\n{UNSTRUCTURED_ROOT}")

    supported_extensions = {".pdf", ".docx", ".txt"}

    files = sorted(
        path
        for path in UNSTRUCTURED_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    print(f"\nDocuments found: {len(files)}")

    if not files:
        print("No PDF/DOCX/TXT files found.")
        return

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    embedding_dimension = model.get_sentence_embedding_dimension()

    print("Embedding model loaded.")
    print("Embedding dimensions:", embedding_dimension)

    if embedding_dimension != 384:
        raise ValueError(
            f"Expected 384-dimensional embeddings, got "
            f"{embedding_dimension}."
        )

    print("\nConnecting to MongoDB Atlas...")

    client = MongoClient(MONGO_URI)

    try:
        client.admin.command("ping")
        print("Connected to MongoDB Atlas successfully.")

        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        # Prevent duplicate chunks with the same stable chunk_id.
        collection.create_index(
            "chunk_id",
            unique=True,
            name="unique_chunk_id"
        )

        # Helpful for document-level operations.
        collection.create_index(
            "document_id",
            name="document_id_index"
        )

        # Track exact duplicate source documents during this run.
        seen_content_hashes = {}
        total_chunks = 0
        skipped_duplicates = 0
        successful_documents = 0
        failed_documents = 0

        for file_path in files:

            print("\n" + "-" * 70)
            print(f"Processing: {file_path.name}")

            try:
                # ----------------------------------------------------
                # Extract + clean
                # ----------------------------------------------------

                text = extract_text(file_path)
                text = clean_text(text)

                if not text:
                    print("WARNING: No text extracted. Skipping.")
                    continue

                content_hash = sha256_text(text)

                # ----------------------------------------------------
                # Detect exact duplicate source documents
                # ----------------------------------------------------

                if content_hash in seen_content_hashes:
                    original = seen_content_hashes[content_hash]

                    print(
                        "WARNING: Exact duplicate content detected."
                    )
                    print(f"Original: {original}")
                    print(f"Duplicate: {file_path}")

                    skipped_duplicates += 1
                    continue

                seen_content_hashes[content_hash] = str(file_path)

                print(f"Extracted characters: {len(text)}")

                # ----------------------------------------------------
                # Metadata
                # ----------------------------------------------------

                metadata = get_metadata(file_path, text)

                document_id = create_document_id(
                    file_path,
                    metadata
                )

                print(f"Document ID: {document_id}")
                print(
                    f"Domain: {metadata.get('domain')}"
                )
                print(
                    f"Document type: {metadata.get('document_type')}"
                )
                print(
                    f"Category: {metadata.get('category')}"
                )

                # ----------------------------------------------------
                # Chunk
                # ----------------------------------------------------

                chunks = chunk_text(
                    text,
                    chunk_size=500,
                    overlap=100
                )

                print(f"Chunks created: {len(chunks)}")

                if not chunks:
                    print("WARNING: No chunks created. Skipping.")
                    continue

                # ----------------------------------------------------
                # Embeddings
                # ----------------------------------------------------

                embeddings = model.encode(
                    chunks,
                    show_progress_bar=False
                )

                # ----------------------------------------------------
                # Build MongoDB documents
                # ----------------------------------------------------

                documents = []

                for index, (chunk, embedding) in enumerate(
                    zip(chunks, embeddings)
                ):

                    chunk_id = (
                        f"{document_id}"
                        f"::chunk-{index + 1:04d}"
                    )

                    document = {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "content_hash": sha256_text(chunk),
                        "embedding": embedding.tolist(),

                        "product": metadata.get("product"),
                        "domain": metadata.get("domain"),
                        "department": metadata.get("department"),
                        "document_type": metadata.get("document_type"),
                        "category": metadata.get("category"),

                        "revision": metadata.get("revision"),
                        "status": metadata.get("status"),

                        "file_name": metadata["file_name"],
                        "relative_path": metadata["relative_path"],

                        "source_type": "synthetic",
                        "embedding_model": MODEL_NAME,
                        "embedding_dimension": embedding_dimension,

                        "chunk_index": index,
                        "chunk_size": len(chunk.split()),
                    }

                    documents.append(document)

                # ----------------------------------------------------
                # Idempotent replacement for this document
                # ----------------------------------------------------
                #
                # We first remove this document's old chunks.
                # This handles the case where a document was previously
                # ingested with a different number of chunks.
                # ----------------------------------------------------

                collection.delete_many(
                    {"document_id": document_id}
                )

                if documents:
                    collection.insert_many(
                        documents,
                        ordered=True
                    )

                    inserted_count = len(documents)
                    total_chunks += inserted_count
                    successful_documents += 1

                    print(
                        f"Inserted/replaced chunks: "
                        f"{inserted_count}"
                    )

            except Exception as e:
                failed_documents += 1

                print(
                    f"ERROR processing {file_path.name}: {e}"
                )

        # ------------------------------------------------------------
        # Final validation
        # ------------------------------------------------------------

        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)

        print(f"Source documents found: {len(files)}")
        print(
            f"Successfully ingested: "
            f"{successful_documents}"
        )
        print(
            f"Exact duplicate source files skipped: "
            f"{skipped_duplicates}"
        )
        print(
            f"Failed documents: "
            f"{failed_documents}"
        )
        print(
            f"Chunks inserted/replaced this run: "
            f"{total_chunks}"
        )

        mongo_count = collection.count_documents({})
        print(
            f"Total chunks currently in MongoDB: "
            f"{mongo_count}"
        )

        print("=" * 70)
        print("UNSTRUCTURED DATA INGESTION COMPLETED")
        print("=" * 70)

    finally:
        client.close()
        print("\nMongoDB connection closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
