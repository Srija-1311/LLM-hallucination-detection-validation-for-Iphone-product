"""Sweep embedding models and chunk strategies; report Recall@5 and MRR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import REPO_ROOT
from app.services.chunking import chunk_document, parse_header_metadata
from app.services.dataset import resolve_dataset_root
from app.services.ingestion_service import bucket_for_path, extract_text, iter_source_files

PROBE_PATH = REPO_ROOT / "data" / "retrieval_probe.json"

MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
]
STRATEGIES = ["table_row", "table_section", "section", "word_200", "word_400"]


def _prefix(model_name: str, text: str, kind: str) -> str:
    if "e5-" in model_name.lower():
        return f"{kind}: {text}"
    return text


def evaluate(model_name: str, strategy: str, probes: list[dict], files: list) -> dict:
    model = SentenceTransformer(model_name)
    corpus_texts: list[str] = []
    corpus_docs: list[str] = []

    for path, bucket in files:
        text = extract_text(path)
        if not text.strip():
            continue
        metadata = parse_header_metadata(text, path, bucket)
        chunks = chunk_document(text, metadata, strategy=strategy)
        for chunk in chunks:
            corpus_texts.append(_prefix(model_name, chunk["content"], "passage"))
            corpus_docs.append(metadata["document_id"])

    if not corpus_texts:
        return {"model": model_name, "strategy": strategy, "recall_at_5": 0.0, "mrr": 0.0, "chunks": 0}

    embeddings = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    hits = 0
    reciprocal_ranks = []
    failures = []

    for probe in probes:
        query = _prefix(model_name, probe["query"], "query")
        query_vec = model.encode(query, normalize_embeddings=True)
        scores = embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:5]
        retrieved = [corpus_docs[i] for i in top_idx]
        relevant = set(probe["relevant_document_ids"])
        rank = None
        for position, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rank = position
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append({
                "query": probe["query"],
                "expected": probe["relevant_document_ids"],
                "retrieved": retrieved,
            })

    n = len(probes)
    return {
        "model": model_name,
        "strategy": strategy,
        "chunks": len(corpus_texts),
        "recall_at_5": hits / n,
        "mrr": float(np.mean(reciprocal_ranks)),
        "failures": failures,
    }


def main() -> None:
    probes = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    dataset_root = resolve_dataset_root()
    files = [
        (path, bucket)
        for path, bucket in iter_source_files(dataset_root)
        if bucket in {"product_specs", "internal_departments", "external_bodies"}
    ]
    results = []
    for model_name in MODELS:
        for strategy in STRATEGIES:
            print(f"Evaluating {model_name} / {strategy} ...")
            result = evaluate(model_name, strategy, probes, files)
            summary = {k: v for k, v in result.items() if k != "failures"}
            print(summary)
            results.append(result)

    ranked = sorted(results, key=lambda item: (item["recall_at_5"], item["mrr"]), reverse=True)
    winner = ranked[0]
    out_path = REPO_ROOT / "data" / "retrieval_tune_results.json"
    out_path.write_text(json.dumps({"winner": winner, "results": ranked}, indent=2), encoding="utf-8")
    print("\nWinner:", {k: winner[k] for k in ("model", "strategy", "recall_at_5", "mrr", "chunks")})
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
