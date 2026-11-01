"""Metrics that return unavailable status when human judgments do not exist."""

from __future__ import annotations

from collections import Counter
from typing import Any


def classification_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("human_label")]
    if not labeled:
        return {"status": "insufficient_labels", "labeled_examples": 0}
    try:
        from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
    except ImportError as error:
        raise RuntimeError("Install scikit-learn to calculate classification metrics.") from error
    y_true = [row["human_label"] for row in labeled]
    y_pred = [row.get("predicted_category", "") for row in labeled]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "status": "preliminary",
        "labeled_examples": len(labeled),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def retrieval_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    if not relevant_ids:
        return {"status": "insufficient_judgments"}  # type: ignore[return-value]
    top = ranked_ids[:k]
    hits = [index for index, value in enumerate(top, start=1) if value in relevant_ids]
    return {
        "precision_at_k": len(hits) / max(1, k),
        "recall_at_k": len(hits) / len(relevant_ids),
        "mrr": 1 / hits[0] if hits else 0.0,
    }
