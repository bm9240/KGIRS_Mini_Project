"""Run the transparent rule-based baseline over genuine citation contexts."""

from pathlib import Path

from src.classification.engine import CitationClassifier
from src.preprocessing.dataset import load_dataset, write_rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    dataset = load_dataset(root)
    classifier = CitationClassifier()
    rows = []
    for row in dataset.contexts:
        prediction = classifier.predict(row)
        rows.append(
            {
                **row,
                "predicted_category": prediction.category,
                "prediction_confidence": str(prediction.confidence),
                "classification_method": prediction.method,
                "classification_evidence": "; ".join(prediction.evidence),
                "human_label": "",
            }
        )
    output = root / "data" / "processed" / "classified_contexts.csv"
    write_rows(output, rows)
    print(f"Classified {len(rows)} contexts with the rule-based baseline.")
    print(f"Output: {output}")
