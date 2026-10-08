from pymongo import MongoClient
from pymongo.database import Database

from app.config import MONGO_DB_NAME, MONGO_URI

PRODUCT_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["product_id", "product_name"],
        "properties": {
            "product_id": {"bsonType": "string"},
            "product_name": {"bsonType": "string"},
            "product_family": {"bsonType": ["string", "null"]},
            "model_year": {"bsonType": ["int", "long", "null"]},
            "synthetic_dataset_flag": {"bsonType": ["bool", "null"]},
            "source_type": {"bsonType": ["string", "null"]},
        },
    }
}

DOCUMENT_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["document_id", "product_id", "bucket"],
        "properties": {
            "document_id": {"bsonType": "string"},
            "product_id": {"bsonType": "string"},
            "bucket": {"bsonType": "string"},
            "status": {"bsonType": ["string", "null"]},
            "file_path": {"bsonType": ["string", "null"]},
        },
    }
}

TRANSACTION_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "transaction_id",
            "product_id",
            "source_table",
            "attribute",
            "canonical_sentence",
        ],
        "properties": {
            "transaction_id": {"bsonType": "string"},
            "product_id": {"bsonType": "string"},
            "source_table": {"bsonType": "string"},
            "attribute": {"bsonType": "string"},
            "canonical_sentence": {"bsonType": "string"},
            "source_document_id": {"bsonType": ["string", "null"]},
            "evidence_status": {"bsonType": ["string", "null"]},
        },
    }
}

AUDIT_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["run_id", "job_type", "started_at", "status"],
        "properties": {
            "run_id": {"bsonType": "string"},
            "job_type": {"bsonType": "string"},
            "status": {"bsonType": "string"},
        },
    }
}

CHUNK_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["chunk_id", "document_id", "content", "embedding"],
        "properties": {
            "chunk_id": {"bsonType": "string"},
            "document_id": {"bsonType": "string"},
            "content": {"bsonType": "string"},
            "embedding": {"bsonType": "array"},
            "bucket": {"bsonType": ["string", "null"]},
            "status": {"bsonType": ["string", "null"]},
            "chunk_kind": {"bsonType": ["string", "null"]},
        },
    }
}


def get_mongo_uri() -> str:
    if not MONGO_URI:
        raise RuntimeError("MONGODB_URI (or MONGO_URI) is missing from the .env file.")
    return MONGO_URI


def get_client() -> MongoClient:
    return MongoClient(get_mongo_uri())


def get_database(client: MongoClient | None = None) -> Database:
    owns_client = client is None
    if owns_client:
        client = get_client()
    return client[MONGO_DB_NAME]


def _ensure_collection(db: Database, name: str, validator: dict, unique_field: str) -> None:
    existing = set(db.list_collection_names())
    if name not in existing:
        db.create_collection(name, validator={"$jsonSchema": validator["$jsonSchema"]})
    else:
        try:
            db.command({
                "collMod": name,
                "validator": {"$jsonSchema": validator["$jsonSchema"]},
                "validationLevel": "moderate",
            })
        except Exception:
            pass

    db[name].create_index([(unique_field, 1)], unique=True, name=f"unique_{unique_field}")


def ensure_collections(db: Database) -> None:
    _ensure_collection(db, "products", PRODUCT_VALIDATOR, "product_id")
    _ensure_collection(db, "documents", DOCUMENT_VALIDATOR, "document_id")
    _ensure_collection(db, "transactions", TRANSACTION_VALIDATOR, "transaction_id")
    _ensure_collection(db, "audit_log", AUDIT_VALIDATOR, "run_id")
    _ensure_collection(db, "document_chunks", CHUNK_VALIDATOR, "chunk_id")
    db["document_chunks"].create_index([("document_id", 1)], name="document_id_index")
    db["document_chunks"].create_index([("bucket", 1)], name="bucket_index")
