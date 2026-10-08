from datetime import datetime, timezone
from uuid import uuid4

from pymongo import UpdateOne

from app.config import MONGO_DB_NAME
from app.models.schemas import AuditLog
from app.services.mongo import ensure_collections, get_client
from app.services.structured_normalize import normalize_structured


def load_structured_mongo() -> dict:
    started = datetime.now(timezone.utc)
    run_id = f"structured-{uuid4().hex[:12]}"
    client = get_client()
    try:
        db = client[MONGO_DB_NAME]
        ensure_collections(db)
        normalized = normalize_structured(write_csv=True)

        products = [item.model_dump() for item in normalized["products"]]
        documents = [item.model_dump() for item in normalized["documents"]]
        transactions = [item.model_dump() for item in normalized["transactions"]]

        if products:
            db["products"].bulk_write(
                [
                    UpdateOne({"product_id": row["product_id"]}, {"$set": row}, upsert=True)
                    for row in products
                ]
            )
        if documents:
            db["documents"].bulk_write(
                [
                    UpdateOne({"document_id": row["document_id"]}, {"$set": row}, upsert=True)
                    for row in documents
                ]
            )
        if transactions:
            db["transactions"].bulk_write(
                [
                    UpdateOne(
                        {"transaction_id": row["transaction_id"]},
                        {"$set": row},
                        upsert=True,
                    )
                    for row in transactions
                ]
            )

        known_products = {row["product_id"] for row in products}
        known_docs = {row["document_id"] for row in documents}
        rejected = []
        for row in transactions:
            if row["product_id"] not in known_products:
                rejected.append({
                    "transaction_id": row["transaction_id"],
                    "error": "unknown product_id",
                })
            doc_id = row.get("source_document_id")
            if doc_id and doc_id not in known_docs:
                rejected.append({
                    "transaction_id": row["transaction_id"],
                    "error": f"unknown source_document_id {doc_id}",
                })

        finished = datetime.now(timezone.utc)
        status = "completed_with_errors" if (normalized["errors"] or rejected) else "success"
        audit = AuditLog(
            run_id=run_id,
            job_type="structured_ingest",
            started_at=started,
            finished_at=finished,
            status=status,
            counts=normalized["counts"],
            errors=normalized["errors"] + rejected,
        )
        db["audit_log"].update_one(
            {"run_id": audit.run_id},
            {"$set": audit.model_dump()},
            upsert=True,
        )
        return {
            "status": status,
            "run_id": run_id,
            "dataset_root": normalized["dataset_root"],
            "counts": normalized["counts"],
            "errors": normalized["errors"] + rejected,
        }
    finally:
        client.close()
