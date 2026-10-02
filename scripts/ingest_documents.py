import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")

MONGO_DB_NAME = os.getenv(
    "MONGO_DB_NAME",
    "foxconn_iphone_kb"
)

MONGO_COLLECTION_NAME = os.getenv(
    "MONGO_COLLECTION_NAME",
    "document_chunks"
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

print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")
print("Embedding dimensions:", model.get_sentence_embedding_dimension())


# ============================================================
# 4. DOCUMENT TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_path):
    """Extract text from a PDF file."""

    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_docx_text(file_path):
    """Extract text from a DOCX file."""

    document = Document(str(file_path))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def extract_txt_text(file_path):
    """Extract text from a TXT file."""

    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def extract_text(file_path):
    """Choose the appropriate extractor based on file type."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    elif extension == ".docx":
        return extract_docx_text(file_path)

    elif extension == ".txt":
        return extract_txt_text(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )


# ============================================================
# 5. TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean unnecessary whitespace."""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 6. CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):
    """
    Split text into word-based chunks.

    chunk_size:
        Approximate number of words per chunk.

    overlap:
        Number of words shared between consecutive chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# 7. METADATA EXTRACTION
# ============================================================

def get_metadata(file_path):
    """
    Derive metadata from the unstructured-data
    folder hierarchy.
    """

    relative_path = file_path.relative_to(
        UNSTRUCTURED_ROOT
    )

    parts = relative_path.parts

    # Example:
    #
    # policies/
    # quality/
    # quality_policy.pdf

    document_type = parts[0] if len(parts) >= 1 else None

    category = parts[1] if len(parts) >= 2 else None

    file_name = file_path.name

    return {
        "document_type": document_type,
        "category": category,
        "file_name": file_name,
        "relative_path": str(relative_path)
    }


# ============================================================
# 8. DOCUMENT ID
# ============================================================

def create_document_id(file_path):
    """
    Create a stable document ID from the file path.
    """

    relative_path = file_path.relative_to(
        UNSTRUCTURED_ROOT
    )

    document_id = str(relative_path)

    document_id = document_id.replace("\\", "/")

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

    print("\n")
    print("=" * 70)
    print("FOXCONN iPHONE UNSTRUCTURED DOCUMENT INGESTION")
    print("=" * 70)

    if not UNSTRUCTURED_ROOT.exists():

        raise FileNotFoundError(
            f"Unstructured dataset not found:\n"
            f"{UNSTRUCTURED_ROOT}"
        )

    print(
        f"\nScanning:\n{UNSTRUCTURED_ROOT}"
    )

    # --------------------------------------------------------
    # Find supported documents
    # --------------------------------------------------------

    supported_extensions = {
        ".pdf",
        ".docx",
        ".txt"
    }

    files = [
        path
        for path in UNSTRUCTURED_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in supported_extensions
    ]

    print(
        f"\nDocuments found: {len(files)}"
    )

    if not files:

        print(
            "No PDF/DOCX/TXT files found."
        )

        return

    # --------------------------------------------------------
    # Connect to MongoDB
    # --------------------------------------------------------

    print("\nConnecting to MongoDB Atlas...")

    client = MongoClient(MONGO_URI)

    client.admin.command("ping")

    print(
        "Connected to MongoDB Atlas successfully."
    )

    db = client[MONGO_DB_NAME]

    collection = db[
        MONGO_COLLECTION_NAME
    ]

    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    total_chunks = 0

    for file_path in files:

        print("\n")
        print("-" * 70)

        print(
            f"Processing: {file_path.name}"
        )

        try:

            # --------------------------------------------
            # Extract text
            # --------------------------------------------

            text = extract_text(file_path)

            text = clean_text(text)

            if not text:

                print(
                    "WARNING: No text extracted. Skipping."
                )

                continue

            print(
                f"Extracted characters: {len(text)}"
            )

            # --------------------------------------------
            # Chunk text
            # --------------------------------------------

            chunks = chunk_text(
                text,
                chunk_size=500,
                overlap=100
            )

            print(
                f"Chunks created: {len(chunks)}"
            )

            # --------------------------------------------
            # Metadata
            # --------------------------------------------

            metadata = get_metadata(
                file_path
            )

            document_id = create_document_id(
                file_path
            )

            # --------------------------------------------
            # Generate embeddings
            # --------------------------------------------

            embeddings = model.encode(
                chunks,
                show_progress_bar=False
            )

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

                    "embedding": embedding.tolist(),

                    "document_type":
                        metadata["document_type"],

                    "category":
                        metadata["category"],

                    "file_name":
                        metadata["file_name"],

                    "relative_path":
                        metadata["relative_path"],

                    "source_type":
                        "synthetic",

                    "embedding_model":
                        MODEL_NAME,

                    "chunk_index":
                        index,

                    "chunk_size":
                        len(chunk.split())
                }

                documents.append(document)

            # --------------------------------------------
            # Insert into MongoDB
            # --------------------------------------------

            if documents:

                result = collection.insert_many(
                    documents
                )

                inserted_count = len(
                    result.inserted_ids
                )

                total_chunks += inserted_count

                print(
                    f"Inserted chunks: "
                    f"{inserted_count}"
                )

        except Exception as e:

            print(
                f"ERROR processing "
                f"{file_path.name}: {e}"
            )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "UNSTRUCTURED DATA INGESTION COMPLETED"
    )

    print(
        f"Total chunks inserted: {total_chunks}"
    )

    print("=" * 70)

    client.close()

    print(
        "\nMongoDB connection closed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()