"""Rule and optional supervised citation-intent classifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CATEGORIES = (
    "SUPPORTING_OR_RELIANCE",
    "BACKGROUND",
    "METHODOLOGICAL",
    "CRITICAL",
    "RETRACTION_ACKNOWLEDGEMENT",
    "UNCLEAR",
)

RULES = {
    "RETRACTION_ACKNOWLEDGEMENT": (
        "retracted", "retraction", "withdrawn", "withdrawal", "retract"
    ),
    "CRITICAL": (
        "however", "disagree", "contrary", "failed", "limitation", "question",
        "inconsistent", "critici", "not support",
    ),
    "METHODOLOGICAL": (
        "method", "protocol", "algorithm", "procedure", "technique", "assay",
        "measurement",
    ),
    "SUPPORTING_OR_RELIANCE": (
        "demonstrated", "consistent with", "as reported", "according to",
        "previously showed", "used the findings", "based on",
    ),
    "BACKGROUND": (
        "review", "reported", "described", "background", "previous studies",
        "literature",
    ),
}


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    evidence: tuple[str, ...]
    method: str


class RuleBasedClassifier:
    """Conservative baseline: conflicting or weak signals remain UNCLEAR."""

    def predict(self, sentence: str, paragraph: str = "") -> Prediction:
        text = f"{sentence} {paragraph}".lower()
        scores: dict[str, int] = {}
        evidence: dict[str, list[str]] = {}
        for category, terms in RULES.items():
            hits = [term for term in terms if re.search(r"\b" + re.escape(term), text)]
            if hits:
                scores[category] = len(hits)
                evidence[category] = hits
        if not scores:
            return Prediction("UNCLEAR", 0.2, (), "rule_based")
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return Prediction("UNCLEAR", 0.35, tuple(evidence[ranked[0][0]]), "rule_based")
        category, score = ranked[0]
        confidence = min(0.95, 0.5 + 0.1 * score)
        return Prediction(category, confidence, tuple(evidence[category]), "rule_based")


class CitationClassifier:
    """Reusable classifier facade with explicit optional model backends."""

    def __init__(self) -> None:
        self.rule_based = RuleBasedClassifier()
        self.supervised_model: Any | None = None
        self.supervised_vectorizer: Any | None = None

    def predict(self, row: dict[str, str], method: str = "rule_based") -> Prediction:
        sentence = row.get("citation_sentence", "")
        paragraph = row.get("paragraph_text", "")
        if method == "rule_based":
            return self.rule_based.predict(sentence, paragraph)
        if self.supervised_model is None or self.supervised_vectorizer is None:
            raise RuntimeError(
                "A supervised classifier is not trained. Provide human labels first."
            )
        features = self.supervised_vectorizer.transform([f"{sentence} {paragraph}"])
        probabilities = self.supervised_model.predict_proba(features)[0]
        index = probabilities.argmax()
        return Prediction(
            str(self.supervised_model.classes_[index]),
            float(probabilities[index]),
            (),
            method,
        )

    def fit_tfidf_logistic(self, rows: list[dict[str, str]]) -> None:
        labeled = [row for row in rows if row.get("human_label") in CATEGORIES]
        if len(set(row["human_label"] for row in labeled)) < 2:
            raise ValueError("At least two human-labeled categories are required.")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:
            raise RuntimeError("Install scikit-learn to train the TF-IDF model.") from error
        texts = [f"{row.get('citation_sentence', '')} {row.get('paragraph_text', '')}" for row in labeled]
        self.supervised_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        features = self.supervised_vectorizer.fit_transform(texts)
        self.supervised_model = LogisticRegression(max_iter=1000).fit(
            features, [row["human_label"] for row in labeled]
        )
