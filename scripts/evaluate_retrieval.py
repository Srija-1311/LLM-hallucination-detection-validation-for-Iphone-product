"""
Evaluate retrieval quality against a labelled retrieval benchmark.

Metrics:
- Recall@1
- Recall@3
- Recall@5
- MRR
- nDCG@5

This script evaluates retrieval locally using the same chunking
pipeline used by the project. It does NOT require MongoDB Atlas.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.chunking import chunk_document, parse_header_metadata
from app.services.dataset import resolve_dataset_root
from app.services.ingestion_service import (
    bucket_for_path,
    extract_text,
    iter_source_files,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EVAL_PATH = (
    ROOT
    / "experiments"
    / "retrieval"
    / "datasets"
    / "retrieval_eval.json"
)

RESULT_DIR = (
    ROOT
    / "experiments"
    / "retrieval"
    / "results"
)

MODEL_NAME = "all-MiniLM-L6-v2"

# Start with the existing strategy.
# We will experiment with this later.
CHUNK_STRATEGY = "table_section"

TOP_K = 5


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def e5_prefix(model_name: str, text: str, kind: str) -> str:
    """
    E5 models require query:/passage: prefixes.
    MiniLM/BGE don't require them.
    """
    if "e5-" in model_name.lower():
        return f"{kind}: {text}"

    return text


def dcg(relevances: list[int]) -> float:
    """
    Discounted cumulative gain.
    """
    score = 0.0

    for rank, relevance in enumerate(relevances, start=1):
        if relevance:
            score += relevance / math.log2(rank + 1)

    return score


def ndcg_at_k(
    retrieved_documents: list[str],
    relevant_documents: set[str],
    k: int,
) -> float:

    retrieved = retrieved_documents[:k]

    relevances = [
        1 if doc_id in relevant_documents else 0
        for doc_id in retrieved
    ]

    actual_dcg = dcg(relevances)

    ideal_relevances = [1] * min(len(relevant_documents), k)

    ideal_dcg = dcg(ideal_relevances)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ---------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------

def build_corpus(model_name: str, strategy: str):
    """
    Read the project's source documents and create chunks.

    Returns:
        corpus_texts
        corpus_metadata
    """

    dataset_root = resolve_dataset_root()

    files = [
        (path, bucket)
        for path, bucket in iter_source_files(dataset_root)
        if bucket in {
            "product_specs",
            "internal_departments",
            "external_bodies",
            "unstructured",
        }
    ]

    corpus_texts = []
    corpus_metadata = []

    for path, bucket in files:

        text = extract_text(path)

        if not text.strip():
            continue

        metadata = parse_header_metadata(
            text,
            path,
            bucket,
        )

        chunks = chunk_document(
            text,
            metadata,
            strategy=strategy,
        )

        for chunk in chunks:

            corpus_texts.append(
                e5_prefix(
                    model_name,
                    chunk["content"],
                    "passage",
                )
            )

            corpus_metadata.append(
                {
                    "document_id": metadata["document_id"],
                    "bucket": bucket,
                    "chunk_kind": chunk.get("chunk_kind"),
                    "content": chunk["content"],
                }
            )

    return corpus_texts, corpus_metadata


# ---------------------------------------------------------
# Document-level ranking
# ---------------------------------------------------------

def rank_documents(
    scores: np.ndarray,
    corpus_metadata: list[dict],
    top_k: int,
):
    """
    Convert chunk-level scores into document-level ranking.

    If multiple chunks belong to the same document,
    only the highest-scoring chunk is used for that document.

    This prevents one document from occupying multiple
    positions in the top-k.
    """

    sorted_indices = np.argsort(scores)[::-1]

    best_document_score = {}

    for index in sorted_indices:

        metadata = corpus_metadata[index]

        document_id = metadata["document_id"]

        score = float(scores[index])

        if (
            document_id not in best_document_score
            or score > best_document_score[document_id]
        ):
            best_document_score[document_id] = score

    ranked = sorted(
        best_document_score.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked[:top_k]


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate():

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Model       : {MODEL_NAME}")
    print(f"Chunking    : {CHUNK_STRATEGY}")
    print(f"Evaluation  : {EVAL_PATH}")

    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVAL_PATH}"
        )

    probes = json.loads(
        EVAL_PATH.read_text(encoding="utf-8")
    )

    print(f"Queries     : {len(probes)}")

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------------------------------
    # Build corpus
    # -----------------------------------------------------

    print("Building corpus...")

    corpus_texts, corpus_metadata = build_corpus(
        MODEL_NAME,
        CHUNK_STRATEGY,
    )

    print(f"Corpus chunks: {len(corpus_texts)}")

    if not corpus_texts:
        raise RuntimeError(
            "No corpus chunks were created."
        )

    # -----------------------------------------------------
    # Embed corpus
    # -----------------------------------------------------

    print("\nEncoding corpus...")

    corpus_embeddings = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    recall_1_hits = 0
    recall_3_hits = 0
    recall_5_hits = 0

    reciprocal_ranks = []

    ndcg_scores = []

    failures = []

    # -----------------------------------------------------
    # Evaluate each query
    # -----------------------------------------------------

    for probe in probes:

        query = probe["query"]

        relevant_documents = set(
            probe["expected_document_ids"]
        )

        query_text = e5_prefix(
            MODEL_NAME,
            query,
            "query",
        )

        query_embedding = model.encode(
            query_text,
            normalize_embeddings=True,
        )

        scores = corpus_embeddings @ query_embedding

        ranked_documents = rank_documents(
            scores,
            corpus_metadata,
            TOP_K,
        )

        retrieved_documents = [
            document_id
            for document_id, score in ranked_documents
        ]

        # ---------------------------------------------
        # Recall@K
        # ---------------------------------------------

        if any(
            doc in relevant_documents
            for doc in retrieved_documents[:1]
        ):
            recall_1_hits += 1

        if any(
            doc in relevant_documents
            for doc in retrieved_documents[:3]
        ):
            recall_3_hits += 1

        if any(
            doc in relevant_documents
            for doc in retrieved_documents[:5]
        ):
            recall_5_hits += 1

        # ---------------------------------------------
        # MRR
        # ---------------------------------------------

        reciprocal_rank = 0.0

        for rank, document_id in enumerate(
            retrieved_documents,
            start=1,
        ):

            if document_id in relevant_documents:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(
            reciprocal_rank
        )

        # ---------------------------------------------
        # nDCG@5
        # ---------------------------------------------

        ndcg_scores.append(
            ndcg_at_k(
                retrieved_documents,
                relevant_documents,
                TOP_K,
            )
        )

        # ---------------------------------------------
        # Store failures
        # ---------------------------------------------

        if reciprocal_rank == 0:

            failures.append(
                {
                    "query_id": probe["query_id"],
                    "query": query,
                    "expected": list(
                        relevant_documents
                    ),
                    "retrieved": retrieved_documents,
                }
            )

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    total = len(probes)

    results = {
        "model": MODEL_NAME,
        "chunk_strategy": CHUNK_STRATEGY,
        "top_k": TOP_K,
        "queries": total,
        "corpus_chunks": len(corpus_texts),
        "recall_at_1": recall_1_hits / total,
        "recall_at_3": recall_3_hits / total,
        "recall_at_5": recall_5_hits / total,
        "mrr": float(
            np.mean(reciprocal_ranks)
        ),
        "ndcg_at_5": float(
            np.mean(ndcg_scores)
        ),
        "failures": failures,
    }

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Recall@1 : {results['recall_at_1']:.4f}"
    )

    print(
        f"Recall@3 : {results['recall_at_3']:.4f}"
    )

    print(
        f"Recall@5 : {results['recall_at_5']:.4f}"
    )

    print(
        f"MRR      : {results['mrr']:.4f}"
    )

    print(
        f"nDCG@5   : {results['ndcg_at_5']:.4f}"
    )

    print(
        f"Failures : {len(failures)}"
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULT_DIR
        / "baseline_minilm_table_section.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )


if __name__ == "__main__":
    evaluate()