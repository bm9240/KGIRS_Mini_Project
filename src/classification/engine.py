"""Rule and optional supervised citation-intent classifiers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATEGORIES = (
    "SUPPORTING_OR_RELIANCE",
    "BACKGROUND",
    "METHODOLOGICAL",
    "CRITICAL",
    "RETRACTION_ACKNOWLEDGEMENT",
    "UNCLEAR",
)

# High-precision keywords and multi-word phrases
RULES: dict[str, tuple[str, ...]] = {
    "RETRACTION_ACKNOWLEDGEMENT": (
        "retracted", "retraction", "retracted article", "notice of retraction",
        "withdrawn", "withdrawal", "expression of concern", "editorial notice",
        "retract",
    ),
    "CRITICAL": (
        "contradict", "contradicts", "contradicted", "in contrast to",
        "failed to replicate", "could not replicate", "inconsistent with",
        "disagree", "disagrees", "disputed", "refuted", "refute",
        "flawed", "erroneous", "invalid", "unreliable", "limitation of",
        "questioned the validity", "not supported by", "critique", "criticism",
    ),
    "METHODOLOGICAL": (
        "protocol", "method", "methodology", "procedure", "algorithm",
        "technique", "assay", "measurement", "synthesized according to",
        "adopted the method", "using the protocol", "procedure described",
        "prepared according to", "experimental design", "instrumentation",
    ),
    "SUPPORTING_OR_RELIANCE": (
        "consistent with", "confirms", "confirmed by", "demonstrated that",
        "demonstrated by", "supported by", "in agreement with", "based on",
        "previously showed", "used the findings", "corroborated by",
        "building on", "substantiated by", "further proved", "evidence supports",
    ),
    "BACKGROUND": (
        "review", "reviewed", "background", "previous studies", "earlier works",
        "literature", "reported that", "described previously", "has been shown",
        "widely studied", "general overview", "as known",
    ),
}

# Strong explicit indicators that immediately confer high category confidence
STRONG_SIGNALS: dict[str, tuple[str, ...]] = {
    "RETRACTION_ACKNOWLEDGEMENT": (
        "retracted", "retraction", "withdrawn", "expression of concern"
    ),
    "CRITICAL": (
        "contradict", "failed to replicate", "in contrast to", "refuted", "flawed"
    ),
    "METHODOLOGICAL": (
        "synthesized according to", "using the protocol", "adopted the method", "assay described"
    ),
    "SUPPORTING_OR_RELIANCE": (
        "in agreement with", "confirms", "corroborated by", "based on the findings"
    ),
}


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    evidence: tuple[str, ...]
    method: str
    sentence_evidence: tuple[str, ...] = ()
    paragraph_evidence: tuple[str, ...] = ()


class RuleBasedClassifier:
    """Evidence-backed rule classifier with contextual sentence vs paragraph weighting."""

    def predict(self, sentence: str, paragraph: str = "") -> Prediction:
        norm_sentence = sentence.lower() if sentence else ""
        norm_paragraph = paragraph.lower() if paragraph else ""
        full_text = f"{norm_sentence} {norm_paragraph}".strip()

        if not full_text:
            return Prediction("UNCLEAR", 0.1, (), "rule_based")

        sentence_hits: dict[str, list[str]] = {}
        paragraph_hits: dict[str, list[str]] = {}
        weighted_scores: dict[str, float] = {}

        # 1. Immediate high-priority check for retraction acknowledgement in sentence
        for term in RULES["RETRACTION_ACKNOWLEDGEMENT"]:
            if re.search(r"\b" + re.escape(term), norm_sentence):
                return Prediction(
                    "RETRACTION_ACKNOWLEDGEMENT",
                    0.95,
                    (f"Sentence mentions retraction: '{term}'",),
                    "rule_based",
                    (term,),
                    (),
                )

        # 2. Match rules with sentence having 2.5x weight over background paragraph
        for category, terms in RULES.items():
            s_hits = [
                term for term in terms
                if re.search(r"\b" + re.escape(term), norm_sentence)
            ]
            p_hits = [
                term for term in terms
                if re.search(r"\b" + re.escape(term), norm_paragraph) and term not in s_hits
            ]
            if s_hits:
                sentence_hits[category] = s_hits
            if p_hits:
                paragraph_hits[category] = p_hits

            # Strong signals get additional bonus
            strong_sentence_bonus = sum(
                1.5 for term in STRONG_SIGNALS.get(category, ())
                if re.search(r"\b" + re.escape(term), norm_sentence)
            )

            score = 2.5 * len(s_hits) + 1.0 * len(p_hits) + strong_sentence_bonus
            if score > 0:
                weighted_scores[category] = score

        if not weighted_scores:
            return Prediction("UNCLEAR", 0.20, (), "rule_based")

        ranked = sorted(weighted_scores.items(), key=lambda item: item[1], reverse=True)
        top_cat, top_score = ranked[0]

        # Check for ambiguity / conflict
        if len(ranked) > 1 and abs(top_score - ranked[1][1]) < 0.5:
            conflicting = [f"{c} ({s:.1f})" for c, s in ranked[:2]]
            evidence_terms = tuple(
                sentence_hits.get(top_cat, []) + paragraph_hits.get(top_cat, [])
            )
            return Prediction(
                "UNCLEAR",
                0.35,
                (f"Conflicting signals: {', '.join(conflicting)}",),
                "rule_based",
                tuple(sentence_hits.get(top_cat, ())),
                tuple(paragraph_hits.get(top_cat, ())),
            )

        # Build evidence list
        evidence_list = []
        if top_cat in sentence_hits:
            evidence_list.append(f"Sentence: {', '.join(sentence_hits[top_cat])}")
        if top_cat in paragraph_hits:
            evidence_list.append(f"Paragraph: {', '.join(paragraph_hits[top_cat])}")

        # Compute calibrated confidence
        if top_cat in sentence_hits:
            base_conf = 0.70
            bonus = min(0.25, 0.08 * (len(sentence_hits[top_cat]) - 1 + top_score / 3.0))
            confidence = round(min(0.95, base_conf + bonus), 2)
        else:
            # Only found in paragraph
            base_conf = 0.50
            bonus = min(0.18, 0.05 * len(paragraph_hits.get(top_cat, [])))
            confidence = round(min(0.68, base_conf + bonus), 2)

        return Prediction(
            top_cat,
            confidence,
            tuple(evidence_list),
            "rule_based",
            tuple(sentence_hits.get(top_cat, ())),
            tuple(paragraph_hits.get(top_cat, ())),
        )


class CitationClassifier:
    """Reusable classifier facade supporting rules, supervised training, and human annotation."""

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
            round(float(probabilities[index]), 2),
            (f"TF-IDF Logistic Regression (p={probabilities[index]:.2f})",),
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


def load_annotation_file(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_human_annotation(
    path: Path,
    annotation_id: str,
    human_label: str,
    annotator: str = "expert_reviewer",
    notes: str = "",
) -> bool:
    """Save or update human annotation in CSV without losing existing data."""
    if not path.is_file():
        return False
    rows = load_annotation_file(path)
    updated = False
    for row in rows:
        if row.get("annotation_id") == str(annotation_id):
            row["human_label"] = human_label
            row["annotator"] = annotator
            row["annotation_notes"] = notes
            updated = True
            break
    if not updated:
        return False
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True
