"""Normalize Foxconn structured CSVs onto product_id, document_id, and sentences."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from app.config import NORMALIZED_DIR, PRODUCT_ID_DEFAULT, PRODUCT_NAME_DEFAULT
from app.models.schemas import KbDocument, Product, Transaction
from app.services.dataset import resolve_dataset_root

DOCUMENT_ID_RE = re.compile(r"^[A-Z]{2,5}-[A-Z0-9_-]+$", re.IGNORECASE)

BUCKETS = {
    "product_specs",
    "internal_departments",
    "external_bodies",
    "unstructured",
    "structured_data",
    "metadata",
}

ATTRIBUTE_TOPIC_ALIASES = {
    "processor": "processor",
    "memory": "memory",
    "display": "display",
    "camera": "camera",
    "main_camera": "camera",
    "battery": "battery",
    "battery_capacity": "battery",
    "wifi": "connectivity",
    "connectivity": "connectivity",
    "design": "design",
    "quality": "quality",
    "claim about processor": "processor",
    "claim about memory": "memory",
    "claim about display": "display",
    "claim about battery": "battery",
    "claim about quality rate": "quality",
    "contradictory synthetic claim": "quality",
    "functional_validation": None,
    "quality_inspection": None,
    "safety_quality": None,
    "validation": None,
}

DEPARTMENT_FALLBACK_DOCS = {
    "qa_qc": "INT-IP17-QAC-002",
    "testing": "INT-IP17-TST-001",
    "manufacturing": "INT-IP17-MFG-001",
    "assembly": "INT-IP17-MFG-001",
    "component supplier": "EXT-IP17-SUP-001",
    "r&d": "PSP-IP17-PRC-001",
    "design_engineering": "PSP-IP17-DES-001",
}

TEST_RESULT_TOPIC = {
    "processor": "processor",
    "memory": "memory",
    "display": "display",
    "camera": "camera",
    "battery": "battery",
    "connectivity": "connectivity",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _looks_like_document_id(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(DOCUMENT_ID_RE.match(value.strip()))


def _topic_key(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def _bucket_from_domain(domain: Optional[str], file_path: Optional[str]) -> str:
    domain_key = (domain or "").strip().lower()
    path = (file_path or "").replace("\\", "/")
    first = path.split("/")[0].lower() if path else ""
    if domain_key in BUCKETS:
        return domain_key
    if first in BUCKETS:
        return first
    if domain_key in {"policies", "testing", "manufacturing", "quality", "logistics", "procurement"}:
        return "unstructured"
    if first in {"policies", "testing", "manufacturing", "quality", "logistics", "procurement", "approvals"}:
        return "unstructured"
    return "unstructured"


def _normalize_unstructured_path(file_path: Optional[str]) -> Optional[str]:
    path = _clean(file_path)
    if not path:
        return path
    path = path.replace("\\", "/")
    if path.startswith("unstructured/"):
        return path
    if path.split("/")[0] in {
        "policies",
        "testing",
        "manufacturing",
        "quality",
        "logistics",
        "procurement",
        "approvals",
    }:
        return f"unstructured/{path}"
    return path


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_products(dataset_root: Path) -> list[Product]:
    path = dataset_root / "structured_data" / "product_master" / "product_master.csv"
    rows = _read_csv(path)
    fields: dict[str, str] = {}
    for row in rows:
        attribute = _topic_key(row.get("attribute_or_type"))
        value = _clean(row.get("value"))
        if attribute and value:
            fields[attribute] = value

    product_id = fields.get("product_id", PRODUCT_ID_DEFAULT)
    year_raw = fields.get("model_year")
    model_year = None
    if year_raw:
        try:
            model_year = int(float(year_raw))
        except ValueError:
            model_year = None

    flag_raw = (fields.get("synthetic_dataset_flag") or "true").lower()
    product = Product(
        product_id=product_id,
        product_name=PRODUCT_NAME_DEFAULT,
        product_family=fields.get("product_family", "iPhone"),
        model_year=model_year,
        synthetic_dataset_flag=flag_raw in {"true", "yes", "1"},
        source_type="synthetic",
    )
    return [product]


def _product_id_for_name(name: Optional[str], products: list[Product]) -> str:
    text = (name or "").strip().lower()
    for product in products:
        if text in {product.product_id.lower(), product.product_name.lower()}:
            return product.product_id
        if "iphone 17" in text or text == "ip17":
            return product.product_id
    raise ValueError(f"Product '{name}' is not in the products master.")


def build_documents(dataset_root: Path, products: list[Product]) -> list[KbDocument]:
    documents: dict[str, KbDocument] = {}

    registry_path = dataset_root / "metadata" / "document_registry_final.csv"
    for row in _read_csv(registry_path):
        document_id = _clean(row.get("document_id"))
        if not document_id:
            continue
        product_id = _product_id_for_name(row.get("product"), products)
        file_path = _normalize_unstructured_path(row.get("file_path"))
        documents[document_id] = KbDocument(
            document_id=document_id,
            product_id=product_id,
            product_name=_clean(row.get("product")),
            bucket=_bucket_from_domain(row.get("domain"), file_path),
            domain=_clean(row.get("domain")),
            organization=_clean(row.get("organization")),
            department=_clean(row.get("department")),
            document_type=_clean(row.get("document_type")),
            revision=_clean(row.get("revision")),
            status="active",
            topic=_clean(row.get("topic")),
            file_path=file_path,
            source_type=_clean(row.get("source_type")) or "synthetic",
            confidentiality=_clean(row.get("confidentiality")),
            date=_clean(row.get("date")),
        )

    unstructured_meta = (
        dataset_root / "unstructured" / "unstructured_document_metadata.csv"
    )
    if unstructured_meta.exists():
        for row in _read_csv(unstructured_meta):
            document_id = _clean(row.get("document_id"))
            if not document_id:
                continue
            file_path = _normalize_unstructured_path(row.get("file_path"))
            product_id = _product_id_for_name(row.get("product"), products)
            documents.setdefault(
                document_id,
                KbDocument(
                    document_id=document_id,
                    product_id=product_id,
                    product_name=_clean(row.get("product")),
                    bucket="unstructured",
                    domain=_clean(row.get("domain")),
                    organization=_clean(row.get("organization")),
                    department=_clean(row.get("department")),
                    document_type=_clean(row.get("document_type")),
                    revision=_clean(row.get("revision")),
                    status=_clean(row.get("status")) or "active_synthetic",
                    topic=_clean(row.get("topic")),
                    file_path=file_path,
                    source_type=_clean(row.get("source_type")) or "synthetic",
                    confidentiality=_clean(row.get("confidentiality")),
                    date=_clean(row.get("effective_date")),
                ),
            )

    return list(documents.values())


def _traceability_map(dataset_root: Path) -> dict[str, str]:
    path = dataset_root / "structured_data" / "traceability" / "traceability.csv"
    mapping: dict[str, str] = {}
    for row in _read_csv(path):
        topic = _topic_key(row.get("attribute_or_type"))
        document_id = _clean(row.get("value"))
        if topic and document_id:
            mapping[topic] = document_id
    return mapping


def resolve_source_document_id(
    source_reference: Optional[str],
    attribute: Optional[str],
    unit: Optional[str],
    source_table: str,
    known_ids: set[str],
    topic_to_doc: dict[str, str],
) -> Optional[str]:
    reference = _clean(source_reference)
    if reference and _looks_like_document_id(reference):
        if reference not in known_ids:
            raise ValueError(
                f"source_reference '{reference}' looks like a document_id but is not in documents."
            )
        return reference

    candidates = [
        ATTRIBUTE_TOPIC_ALIASES.get((attribute or "").strip().lower()),
        _topic_key(attribute),
        _topic_key(unit),
        _topic_key(reference),
        TEST_RESULT_TOPIC.get((reference or "").strip().lower()),
    ]
    for key in candidates:
        if key and key in topic_to_doc:
            return topic_to_doc[key]

    dept_key = (reference or "").strip().lower()
    if dept_key in DEPARTMENT_FALLBACK_DOCS:
        fallback = DEPARTMENT_FALLBACK_DOCS[dept_key]
        if fallback in known_ids:
            return fallback

    if source_table == "claim_evidence" and _topic_key(unit) in topic_to_doc:
        return topic_to_doc[_topic_key(unit)]

    return None


def _canonical_sentence(
    product_name: str,
    attribute: str,
    value: Optional[str],
    unit: Optional[str],
    source_document_id: Optional[str],
    source_table: str,
    evidence_status: Optional[str],
) -> str:
    attr = attribute.replace("_", " ")
    unit_text = ""
    if unit and unit not in {"-", "boolean", "synthetic"}:
        if value and unit.lower() not in value.lower():
            unit_text = f" {unit}"
    citation = f" [{source_document_id}]" if source_document_id else ""

    if source_table == "claim_evidence":
        status = evidence_status or value or "UNKNOWN"
        return f"{product_name} {attr} is {status} for {unit or 'unspecified topic'}{citation}."

    if source_table == "traceability":
        return f"{product_name} {attr} evidence document is {value}{citation}."

    return f"{product_name} {attr} is {value or 'unspecified'}{unit_text}{citation}."


def build_transactions(
    dataset_root: Path,
    products: list[Product],
    documents: list[KbDocument],
) -> tuple[list[Transaction], list[dict[str, str]]]:
    known_ids = {doc.document_id for doc in documents}
    topic_to_doc = _traceability_map(dataset_root)
    product_name_by_id = {item.product_id: item.product_name for item in products}
    errors: list[dict[str, str]] = []
    transactions: list[Transaction] = []

    tables = {
        "product_master": dataset_root / "structured_data" / "product_master" / "product_master.csv",
        "component_specs": dataset_root / "structured_data" / "component_specs" / "component_specs.csv",
        "claim_evidence": dataset_root / "structured_data" / "claim_evidence" / "claim_evidence.csv",
        "defect_records": dataset_root / "structured_data" / "defect_records" / "defect_records.csv",
        "manufacturing_records": dataset_root / "structured_data" / "manufacturing_records" / "manufacturing_records.csv",
        "quality_metrics": dataset_root / "structured_data" / "quality_metrics" / "quality_metrics.csv",
        "supplier_records": dataset_root / "structured_data" / "supplier_records" / "supplier_records.csv",
        "test_results": dataset_root / "structured_data" / "test_results" / "test_results.csv",
        "traceability": dataset_root / "structured_data" / "traceability" / "traceability.csv",
    }

    for source_table, path in tables.items():
        if not path.exists():
            errors.append({"file": str(path), "error": "missing structured csv"})
            continue
        for row in _read_csv(path):
            try:
                product_id = _product_id_for_name(row.get("product"), products)
                attribute = _clean(row.get("attribute_or_type")) or "unknown"
                value = _clean(row.get("value"))
                unit = _clean(row.get("unit_or_status"))
                evidence_status = None
                if source_table == "claim_evidence":
                    evidence_status = value
                source_document_id = resolve_source_document_id(
                    row.get("source_reference"),
                    attribute,
                    unit,
                    source_table,
                    known_ids,
                    topic_to_doc,
                )
                sentence = _canonical_sentence(
                    product_name_by_id.get(product_id, PRODUCT_NAME_DEFAULT),
                    attribute,
                    value,
                    unit,
                    source_document_id,
                    source_table,
                    evidence_status,
                )
                transactions.append(
                    Transaction(
                        transaction_id=_clean(row.get("record_id")) or f"{source_table}-{attribute}",
                        product_id=product_id,
                        source_table=source_table,
                        attribute=attribute,
                        value=value,
                        unit=unit,
                        source_document_id=source_document_id,
                        evidence_status=evidence_status,
                        canonical_sentence=sentence,
                        raw=dict(row),
                    )
                )
            except Exception as exc:
                errors.append({
                    "file": str(path),
                    "record_id": str(row.get("record_id")),
                    "error": str(exc),
                })

    return transactions, errors


def normalize_structured(write_csv: bool = True) -> dict[str, Any]:
    dataset_root = resolve_dataset_root()
    products = build_products(dataset_root)
    documents = build_documents(dataset_root, products)
    transactions, errors = build_transactions(dataset_root, products, documents)

    if write_csv:
        NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
        _write_csv(
            NORMALIZED_DIR / "products.csv",
            [item.model_dump() for item in products],
            ["product_id", "product_name", "product_family", "model_year", "synthetic_dataset_flag", "source_type"],
        )
        _write_csv(
            NORMALIZED_DIR / "documents.csv",
            [item.model_dump() for item in documents],
            [
                "document_id",
                "product_id",
                "product_name",
                "bucket",
                "domain",
                "organization",
                "department",
                "document_type",
                "revision",
                "status",
                "topic",
                "file_path",
                "source_type",
                "confidentiality",
                "date",
            ],
        )
        _write_csv(
            NORMALIZED_DIR / "transactions.csv",
            [item.model_dump() for item in transactions],
            [
                "transaction_id",
                "product_id",
                "source_table",
                "attribute",
                "value",
                "unit",
                "source_document_id",
                "evidence_status",
                "canonical_sentence",
            ],
        )

    linked = sum(1 for item in transactions if item.source_document_id)
    return {
        "dataset_root": str(dataset_root),
        "products": products,
        "documents": documents,
        "transactions": transactions,
        "errors": errors,
        "counts": {
            "products": len(products),
            "documents": len(documents),
            "transactions": len(transactions),
            "transactions_with_document": linked,
            "normalize_errors": len(errors),
        },
    }
