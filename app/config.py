import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_ROOT = Path(
    "C:/Users/srija/OneDrive/ドキュメント/"
    "foxconn synthetic data/foxconn synthetic data"
)
DEFAULT_DATASET_ZIP = Path(
    "C:/Users/srija/OneDrive/ドキュメント/"
    "foxconn synthetic data/foxconn synthetic data.zip"
)

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))
DATASET_ZIP = Path(os.getenv("DATASET_ZIP", str(DEFAULT_DATASET_ZIP)))
DATASET_EXTRACT_DIR = Path(
    os.getenv("DATASET_EXTRACT_DIR", str(REPO_ROOT / ".data" / "foxconn_synthetic"))
)
NORMALIZED_DIR = Path(
    os.getenv("NORMALIZED_DIR", str(REPO_ROOT / "data" / "normalized"))
)

MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "foxconn_iphone_kb")
MONGO_CHUNKS_COLLECTION = os.getenv("MONGO_COLLECTION_NAME", "document_chunks")

PRODUCTS_COLLECTION = "products"
DOCUMENTS_COLLECTION = "documents"
TRANSACTIONS_COLLECTION = "transactions"
AUDIT_COLLECTION = "audit_log"

VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

PRODUCT_ID_DEFAULT = "IP17"
PRODUCT_NAME_DEFAULT = "iPhone 17"
