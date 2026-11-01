"""Load the verified Phase 1 outputs without fabricating missing evidence."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import ROOT


@dataclass
class Dataset:
    retracted: list[dict[str, str]]
    citing: list[dict[str, str]]
    contexts: list[dict[str, str]]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dataset(root: Path = ROOT) -> Dataset:
    processed = root / "data" / "processed"
    return Dataset(
        retracted=read_csv(processed / "selected_retracted_papers.csv"),
        citing=read_csv(processed / "citing_papers.csv"),
        contexts=read_csv(processed / "citation_contexts.csv"),
    )


def annotation_rows(dataset: Dataset) -> list[dict[str, str]]:
    """Create an annotation-ready copy with no generated labels."""
    rows = []
    for index, context in enumerate(dataset.contexts, start=1):
        rows.append(
            {
                "annotation_id": str(index),
                **context,
                "predicted_category": "",
                "prediction_confidence": "",
                "human_label": "",
                "annotator": "",
                "annotation_notes": "",
            }
        )
    return rows


def dataset_statistics(dataset: Dataset) -> dict[str, Any]:
    return {
        "retracted_papers": len(dataset.retracted),
        "citing_papers": len(dataset.citing),
        "citation_contexts": len(dataset.contexts),
        "contexts_before_retraction": sum(
            row.get("citation_timing") == "BEFORE_RETRACTION"
            for row in dataset.contexts
        ),
        "contexts_after_retraction": sum(
            row.get("citation_timing") == "AFTER_RETRACTION"
            for row in dataset.contexts
        ),
        "citing_with_pmcid": sum(bool(row.get("pmcid")) for row in dataset.citing),
        "xml_downloaded": sum(
            row.get("xml_status") == "downloaded" for row in dataset.citing
        ),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
