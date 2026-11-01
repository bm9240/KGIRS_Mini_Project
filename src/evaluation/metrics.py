"""Evaluation metrics, benchmarks, and explainability assessment."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from src.classification.engine import CATEGORIES
from src.preprocessing.dataset import Dataset, dataset_statistics


def ndcg_at_k(ranked_ids: list[str], relevance_grades: dict[str, int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank k."""
    if not relevance_grades or k <= 0:
        return 0.0

    # Calculate DCG@k
    top_ids = ranked_ids[:k]
    dcg = 0.0
    for rank_idx, paper_id in enumerate(top_ids, start=1):
        rel = relevance_grades.get(paper_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank_idx + 1)

    # Calculate IDCG@k
    ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for rank_idx, rel in enumerate(ideal_rels, start=1):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(rank_idx + 1)

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def retrieval_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
    relevance_grades: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Calculate Precision@K, Recall@K, MRR, and NDCG@K."""
    if not relevant_ids:
        return {"status": "insufficient_judgments", "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    top = ranked_ids[:k]
    hits = [idx for idx, pid in enumerate(top, start=1) if pid in relevant_ids]
    precision = len(hits) / max(1, k)
    recall = len(hits) / max(1, len(relevant_ids))
    mrr = (1.0 / hits[0]) if hits else 0.0

    grades = relevance_grades or {pid: 1 for pid in relevant_ids}
    ndcg = ndcg_at_k(ranked_ids, grades, k)

    return {
        "k": k,
        "hits": len(hits),
        "total_relevant": len(relevant_ids),
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": ndcg,
    }


def classification_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Calculate classification performance against genuine human-assigned labels."""
    labeled = [row for row in rows if row.get("human_label") and row.get("human_label") in CATEGORIES]
    if len(labeled) < 2:
        return {
            "status": "insufficient_labels",
            "labeled_examples": len(labeled),
            "message": "Human annotations are pending. Use the Citation Intelligence workspace to annotate genuine citation contexts.",
        }

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError as error:
        raise RuntimeError("Install scikit-learn to calculate classification metrics.") from error

    y_true = [row["human_label"] for row in labeled]
    y_pred = [row.get("predicted_category", "UNCLEAR") for row in labeled]

    labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "status": "validated",
        "labeled_examples": len(labeled),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "categories": labels,
        "confusion_matrix": conf_mat,
        "classification_report": report,
    }


# Documented Benchmark Evaluation Queries grounded in the real dataset
EVALUATION_QUERIES = [
    {
        "id": "Q1",
        "query": "ovarian cancer drug resistance cold responsive nanomaterial",
        "target_retracted_id": "70462",
        "description": "Case 1: Nanotechnology drug delivery overcoming ovarian cancer resistance",
    },
    {
        "id": "Q2",
        "query": "miR-129-5p suppresses proliferation migration pancreatic cancer PBX3",
        "target_retracted_id": "66404",
        "description": "Case 2: MicroRNA-129-5p targeting PBX3 in pancreatic cancer cells",
    },
    {
        "id": "Q3",
        "query": "multifunctional cellulose rGO Fe3O4 composite aerogels electromagnetic interference shielding",
        "target_retracted_id": "53194",
        "description": "Case 3: Cellulose graphene oxide composite aerogels for EMI shielding",
    },
    {
        "id": "Q4",
        "query": "cryoablation cancer therapy and nanoparticles mediated cryoablation",
        "target_retracted_id": "70462",
        "description": "Nanoparticle cryoablation therapeutic mechanisms",
    },
    {
        "id": "Q5",
        "query": "apoptosis migration and proliferation regulation in carcinoma cells",
        "target_retracted_id": "66404",
        "description": "Cellular proliferation and apoptosis pathways",
    },
]


def run_retrieval_benchmark(
    dataset: Dataset,
    bm25: Any,
    dense: Any | None,
    hybrid_fn: Callable[[list[list[Any]], int], list[Any]],
    graph_retriever: Any,
    k: int = 5,
) -> dict[str, Any]:
    """Execute evaluation queries and compare BM25, Dense, Hybrid, and Graph-Enhanced retrieval."""
    methods = ["bm25"]
    if dense is not None:
        methods.extend(["dense", "hybrid"])
    methods.append("graph_enhanced")

    results_by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    latencies: dict[str, list[float]] = {m: [] for m in methods}

    for query_spec in EVALUATION_QUERIES:
        query = query_spec["query"]
        target_rid = query_spec["target_retracted_id"]

        # Ground truth relevance:
        # Grade 3: Citing paper with verified context cited AFTER retraction
        # Grade 2: Citing paper with verified context cited BEFORE retraction
        # Grade 1: Citing paper citing the retracted paper in OpenAlex
        relevance_grades: dict[str, int] = {}
        for c in dataset.citing:
            c_id = c.get("openalex_id", "")
            rids = c.get("retracted_record_ids", "").split(";")
            if target_rid in rids:
                relevance_grades[c_id] = 1

        for ctx in dataset.contexts:
            if ctx.get("retracted_record_id") == target_rid:
                c_id = ctx.get("citing_openalex_id", "")
                timing = ctx.get("citation_timing")
                relevance_grades[c_id] = 3 if timing == "AFTER_RETRACTION" else 2

        relevant_ids = set(relevance_grades.keys())

        # 1. BM25
        t0 = time.perf_counter()
        bm25_res = bm25.search(query, top_k=k)
        latencies["bm25"].append((time.perf_counter() - t0) * 1000)
        results_by_method["bm25"].append(
            retrieval_metrics([r.paper_id for r in bm25_res], relevant_ids, k, relevance_grades)
        )

        dense_res = []
        if dense is not None:
            # 2. Dense
            t0 = time.perf_counter()
            dense_res = dense.search(query, top_k=k)
            latencies["dense"].append((time.perf_counter() - t0) * 1000)
            results_by_method["dense"].append(
                retrieval_metrics([r.paper_id for r in dense_res], relevant_ids, k, relevance_grades)
            )

            # 3. Hybrid
            t0 = time.perf_counter()
            hybrid_res = hybrid_fn([bm25_res, dense_res], k)
            latencies["hybrid"].append((time.perf_counter() - t0) * 1000)
            results_by_method["hybrid"].append(
                retrieval_metrics([r.paper_id for r in hybrid_res], relevant_ids, k, relevance_grades)
            )

        # 4. Graph-Enhanced
        t0 = time.perf_counter()
        graph_res = graph_retriever.search(query, top_k=k)
        latencies["graph_enhanced"].append((time.perf_counter() - t0) * 1000)
        results_by_method["graph_enhanced"].append(
            retrieval_metrics([r.paper_id for r in graph_res], relevant_ids, k, relevance_grades)
        )

    # Aggregate metrics across queries
    summary_by_method: dict[str, dict[str, float]] = {}
    for m in methods:
        m_results = results_by_method[m]
        n_queries = max(1, len(m_results))
        avg_precision = sum(r["precision_at_k"] for r in m_results) / n_queries
        avg_recall = sum(r["recall_at_k"] for r in m_results) / n_queries
        avg_mrr = sum(r["mrr"] for r in m_results) / n_queries
        avg_ndcg = sum(r["ndcg_at_k"] for r in m_results) / n_queries
        avg_latency = sum(latencies[m]) / max(1, len(latencies[m]))

        summary_by_method[m] = {
            f"precision@{k}": round(avg_precision, 4),
            f"recall@{k}": round(avg_recall, 4),
            "mrr": round(avg_mrr, 4),
            f"ndcg@{k}": round(avg_ndcg, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }

    return {
        "k": k,
        "total_queries": len(EVALUATION_QUERIES),
        "queries": EVALUATION_QUERIES,
        "summary": summary_by_method,
        "per_query": results_by_method,
    }


def explainability_assessment(dataset: Dataset) -> dict[str, Any]:
    """Perform evidence-completeness audit across extracted citation contexts."""
    contexts = dataset.contexts
    total = len(contexts)
    if total == 0:
        return {"status": "no_contexts", "completeness_rate": 0.0}

    sentence_avail = sum(bool(c.get("citation_sentence", "").strip()) for c in contexts)
    paragraph_avail = sum(bool(c.get("paragraph_text", "").strip()) for c in contexts)
    source_avail = sum(bool(c.get("xml_file", "").strip()) for c in contexts)
    retraction_avail = sum(bool(c.get("retracted_record_id", "").strip()) for c in contexts)
    timing_avail = sum(c.get("citation_timing") in {"BEFORE_RETRACTION", "AFTER_RETRACTION"} for c in contexts)
    ref_marker_avail = sum(bool(c.get("citation_marker", "").strip()) for c in contexts)

    field_rates = {
        "citation_sentence_available": round(sentence_avail / total, 4),
        "surrounding_paragraph_available": round(paragraph_avail / total, 4),
        "source_xml_provenance_available": round(source_avail / total, 4),
        "retraction_metadata_available": round(retraction_avail / total, 4),
        "timing_provenance_available": round(timing_avail / total, 4),
        "citation_marker_available": round(ref_marker_avail / total, 4),
    }

    avg_completeness = sum(field_rates.values()) / len(field_rates)

    return {
        "status": "audited",
        "total_contexts": total,
        "field_rates": field_rates,
        "average_evidence_completeness": round(avg_completeness, 4),
        "methodological_caveat": (
            "Evidence completeness measures presence of verifiable metadata and source full-text excerpts. "
            "It does not assess subjective text comprehension or scientific validity."
        ),
    }


def system_performance_summary(dataset: Dataset, graph_service: Any) -> dict[str, Any]:
    """Summary of system dataset coverage and service health."""
    stats = dataset_statistics(dataset)
    graph_counts = graph_service.counts() if graph_service else {"nodes": 0, "relationships": 0}
    return {
        "dataset_statistics": stats,
        "graph_statistics": graph_counts,
        "neo4j_connected": graph_service.is_connected() if graph_service else False,
        "xml_fulltext_rate": round(stats["xml_downloaded"] / max(1, stats["citing_papers"]), 4),
        "context_extraction_rate": round(stats["citation_contexts"] / max(1, stats["xml_downloaded"]), 4),
    }
