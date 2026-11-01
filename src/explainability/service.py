"""Explanations that preserve the boundary between evidence and inference."""

from __future__ import annotations

from src.classification.engine import CitationClassifier
from src.preprocessing.dataset import Dataset


class ExplanationService:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.classifier = CitationClassifier()

    def context_explanation(self, context: dict[str, str]) -> dict[str, object]:
        prediction = self.classifier.predict(context)
        return {
            "citing_paper_id": context.get("citing_openalex_id", ""),
            "retracted_record_id": context.get("retracted_record_id", ""),
            "citation_sentence": context.get("citation_sentence", ""),
            "paragraph": context.get("paragraph_text", ""),
            "section": context.get("section_heading", ""),
            "intent": prediction.category,
            "confidence": prediction.confidence,
            "classification_method": prediction.method,
            "rule_evidence": list(prediction.evidence),
            "citation_timing": context.get("citation_timing", "DATE_UNCERTAIN"),
            "source": context.get("xml_file", ""),
            "interpretation_note": (
                "The citation is evidence of a scholarly relationship; it does not "
                "prove reliance, validity, or scientific harm."
            ),
        }

    def explain_paper(self, paper_id: str) -> dict[str, object]:
        contexts = [
            row for row in self.dataset.contexts
            if row.get("citing_openalex_id") == paper_id
        ]
        return {
            "paper_id": paper_id,
            "contexts": [self.context_explanation(row) for row in contexts],
            "evidence_status": "verified_contexts" if contexts else "no_extracted_contexts",
        }
