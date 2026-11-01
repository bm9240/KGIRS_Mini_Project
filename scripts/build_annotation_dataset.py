"""Create an annotation-ready copy of genuine extracted citation contexts."""

from pathlib import Path

from src.preprocessing.dataset import annotation_rows, load_dataset, write_rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "annotations" / "citation_context_annotations.csv"
    write_rows(output, annotation_rows(load_dataset(root)))
    print(f"Wrote {output}")
