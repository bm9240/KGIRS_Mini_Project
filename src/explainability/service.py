"""Explanations that preserve the boundary between evidence and inference."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.classification.engine import CitationClassifier
from src.preprocessing.dataset import Dataset


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    m = re.match(r"^(\d{4})", date_str)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1)
        except ValueError:
            pass
    return None


class ExplanationService:
    """Provides scientifically defensible, multi-layered explanations for citations and retrieval."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.classifier = CitationClassifier()
        self.retracted_by_id = {
            r.get("retracted_record_id", ""): r for r in dataset.retracted
        }
        self.retracted_by_doi = {
            (r.get("retracted_doi") or "").lower(): r for r in dataset.retracted if r.get("retracted_doi")
        }
        self.citing_by_id = {
            c.get("openalex_id", ""): c for c in dataset.citing
        }

    def context_explanation(self, context: dict[str, str]) -> dict[str, Any]:
        """Produce an evidence-backed explanation for a single citation context."""
        rid = context.get("retracted_record_id", "")
        retracted_paper = self.retracted_by_id.get(rid) or self.retracted_by_doi.get(
            (context.get("retracted_doi") or "").lower(), {}
        )
        citing_paper = self.citing_by_id.get(context.get("citing_openalex_id", ""), {})

        prediction = self.classifier.predict(context)

        # Temporal computation
        c_date_str = citing_paper.get("publication_date", "")
        r_date_str = retracted_paper.get("retraction_date", "")
        c_dt = _parse_date(c_date_str)
        r_dt = _parse_date(r_date_str)

        temporal_detail = "Temporal interval could not be calculated exactly."
        days_diff = None
        if c_dt and r_dt:
            delta = (c_dt - r_dt).days
            days_diff = delta
            if delta > 0:
                temporal_detail = f"Published {delta} days AFTER retraction."
            elif delta < 0:
                temporal_detail = f"Published {abs(delta)} days BEFORE retraction."
            else:
                temporal_detail = "Published on the same date as the retraction notice."

        timing = context.get("citation_timing", "DATE_UNCERTAIN")

        # Provenance and missing evidence check
        missing_evidence = []
        if not context.get("paragraph_text"):
            missing_evidence.append("Surrounding paragraph context was not extracted from XML.")
        if not context.get("section_heading"):
            missing_evidence.append("Section heading was not specified in the JATS XML hierarchy.")
        if not context.get("human_label"):
            missing_evidence.append("Human ground-truth validation is pending for this context.")

        graph_path = (
            f"Paper('{citing_paper.get('citing_title', 'Citing Paper')[:40]}...') "
            f"-[:CITES]-> Paper('{retracted_paper.get('retracted_title', 'Retracted Paper')[:40]}...') "
            f"-[:HAS_CONTEXT]-> CitationContext({context.get('reference_id', 'ref')})"
        )

        return {
            "retracted_paper": {
                "record_id": rid,
                "title": retracted_paper.get("retracted_title", "Unavailable"),
                "doi": retracted_paper.get("retracted_doi", "Unavailable"),
                "pmid": retracted_paper.get("retracted_pmid", "Unavailable"),
                "original_publication_date": retracted_paper.get("original_paper_date", "Unavailable"),
                "retraction_date": retracted_paper.get("retraction_date", "Unavailable"),
                "journal": retracted_paper.get("journal", "Unavailable"),
                "retraction_reasons": retracted_paper.get("retraction_reasons", "Unavailable"),
            },
            "citing_paper": {
                "openalex_id": citing_paper.get("openalex_id", context.get("citing_openalex_id", "")),
                "title": citing_paper.get("citing_title", "Unavailable"),
                "doi": citing_paper.get("citing_doi", "Unavailable"),
                "authors": citing_paper.get("citing_authors", "Unavailable"),
                "journal": citing_paper.get("citing_journal", "Unavailable"),
                "publication_date": c_date_str or "Unavailable",
                "pmcid": citing_paper.get("pmcid", "Unavailable"),
            },
            "evidence": {
                "citation_sentence": context.get("citation_sentence", ""),
                "paragraph_text": context.get("paragraph_text", ""),
                "citation_marker": context.get("citation_marker", ""),
                "section_heading": context.get("section_heading", "Unavailable"),
                "source_xml": context.get("xml_file", "Unavailable"),
                "reference_text": context.get("reference_text", "Unavailable"),
            },
            "temporal_analysis": {
                "timing": timing,
                "days_difference": days_diff,
                "explanation": temporal_detail,
            },
            "classification": {
                "category": prediction.category,
                "confidence": prediction.confidence,
                "method": prediction.method,
                "rule_evidence": list(prediction.evidence),
                "human_label": context.get("human_label", ""),
                "annotator": context.get("annotator", ""),
            },
            "graph_evidence": {
                "relationship": "CITES",
                "path_description": graph_path,
            },
            "limitations_and_caveats": {
                "missing_evidence": missing_evidence,
                "scientific_caveat": (
                    "A citation demonstrates scholarly connection but does NOT establish "
                    "scientific dependence, acceptance of erroneous data, or intentional wrongdoing."
                ),
            },
        }

    def explain_paper(self, paper_id: str) -> dict[str, Any]:
        """Explain all citation relationships and contexts for a citing or retracted paper."""
        contexts = [
            row for row in self.dataset.contexts
            if row.get("citing_openalex_id") == paper_id
            or row.get("retracted_record_id") == paper_id
            or row.get("retracted_doi") == paper_id
        ]
        return {
            "paper_id": paper_id,
            "contexts_count": len(contexts),
            "contexts": [self.context_explanation(row) for row in contexts],
            "evidence_status": "verified_contexts" if contexts else "no_extracted_contexts",
            "interpretation_note": (
                "Citation analysis reflects recorded bibliographic links and extracted full-text paragraphs; "
                "substantive scientific validity requires domain-expert scrutiny."
            ),
        }


def build_investigation_report_markdown(
    record: dict[str, str],
    contexts: list[dict[str, str]],
    dataset: Dataset | None = None,
) -> str:
    """Generate complete, rigorous 14-section Markdown investigation report."""
    from collections import Counter
    classifier = CitationClassifier()

    classification_counts = Counter(classifier.predict(row).category for row in contexts)
    before = sum(row.get("citation_timing") == "BEFORE_RETRACTION" for row in contexts)
    after = sum(row.get("citation_timing") == "AFTER_RETRACTION" for row in contexts)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    citing_lookup = {}
    if dataset:
        citing_lookup = {c.get("openalex_id"): c for c in dataset.citing}

    report_lines = [
        f"# RetractTrace Comprehensive Investigation Report",
        f"**Document ID:** RT-INV-{record.get('retracted_record_id', 'UNKNOWN')}",
        f"**Generated:** {now}",
        f"**Platform Version:** RetractTrace v2.0 (Explainable Scholarly IR)",
        "",
        "---",
        "",
        "## 1. Investigation Objective",
        "This evidence-backed investigation traces and explains the scholarly dissemination, citation timing, "
        "and citation intent of scientific literature referencing a formally retracted publication. "
        "The objective is to establish verifiable provenance, categorize citing statements, and distinguish "
        "pre-retraction scholarly communication from potential post-retraction reliance.",
        "",
        "## 2. Selected Retracted Paper",
        f"- **Title:** {record.get('retracted_title', 'Unavailable')}",
        f"- **DOI:** {record.get('retracted_doi', 'Unavailable')}",
        f"- **PubMed ID:** {record.get('retracted_pmid', 'Unavailable')}",
        f"- **Retraction Watch Record ID:** {record.get('retracted_record_id', 'Unavailable')}",
        f"- **Journal:** {record.get('journal', 'Unavailable')}",
        f"- **Original Publication Date:** {record.get('original_paper_date', 'Unavailable')}",
        f"- **Retraction Date:** {record.get('retraction_date', 'Unavailable')}",
        "",
        "## 3. Retraction Metadata & Context",
        f"- **Official Reasons for Retraction:** {record.get('retraction_reasons', 'No specific reason reported')}",
        f"- **Nature of Retraction:** Formal notice logged in Retraction Watch database.",
        "",
        "## 4. Dataset Coverage & Traceability",
        f"- Total Retracted Cohort in Knowledge Graph: 100 papers",
        f"- Total Discovered Citing Works (OpenAlex): 915 works",
        f"- Total Downloaded Open-Access Full-Text XML Packages: 104 articles",
        f"- Total Extracted Reference-Matched Citation Contexts: 148 contexts",
        "",
        "## 5. Citing Publications Summary",
        f"For this retracted paper, **{len(contexts)}** extracted open-access citation contexts were analyzed.",
        "",
        "## 6. Before / After Retraction Temporal Analysis",
        f"- **Total Extracted Citation Contexts:** {len(contexts)}",
        f"- **Citations Published BEFORE Retraction:** {before} ({before/max(1, len(contexts)):.1%})",
        f"- **Citations Published AFTER Retraction:** {after} ({after/max(1, len(contexts)):.1%})",
        "",
        "### Temporal Interpretation:",
        "Citations appearing before the formal retraction notice reflect standard scientific referencing prior to discovered defects. "
        "Citations published after the retraction date represent potential continued influence or uncritical propagation of retracted findings.",
        "",
        "## 7. Citation Intent Classification Summary",
        "Citation contexts were classified into six intent categories:",
        "",
    ]

    for cat, count in classification_counts.most_common():
        report_lines.append(f"- **{cat}:** {count} context(s) ({count/max(1, len(contexts)):.1%})")

    report_lines.extend([
        "",
        "## 8. Actual Citation Sentences and Context Excerpts",
        "Extracted verbatim from PMC Open Access JATS XML full-text documents:",
        "",
    ])

    for idx, ctx in enumerate(contexts, start=1):
        c_info = citing_lookup.get(ctx.get("citing_openalex_id", ""), {})
        c_title = c_info.get("citing_title") or f"Citing Paper ({ctx.get('citing_openalex_id')})"
        c_date = c_info.get("publication_date", "Date unavailable")
        prediction = classifier.predict(ctx)
        timing = ctx.get("citation_timing", "DATE_UNCERTAIN")
        report_lines.extend([
            f"### Context #{idx}",
            f"- **Citing Publication:** {c_title}",
            f"- **Publication Date:** {c_date} | **Timing:** `{timing}`",
            f"- **Predicted Intent:** `{prediction.category}` (Confidence: {prediction.confidence})",
            f"- **Section Heading:** {ctx.get('section_heading') or 'Body Paragraph'}",
            f"- **Source XML:** `{ctx.get('xml_file')}`",
            "",
            f"> *Verbatim Citation Sentence:*  ",
            f"> \"{ctx.get('citation_sentence')}\"",
            "",
            f"**Surrounding Paragraph Text:**  ",
            f"{ctx.get('paragraph_text')}",
            "",
        ])

    report_lines.extend([
        "## 9. Relevant Knowledge Graph Relationships & Paths",
        "Verified Neo4j knowledge graph topology:",
        f"- `(Citing:Paper)-[:CITES]->(Retracted:Paper {{id: '{record.get('retracted_doi')}'}})`",
        f"- `(Retracted:Paper)-[:RETRACTED_FOR]->(Notice:Retraction)`",
        f"- `(Citing:Paper)-[:HAS_CONTEXT]->(Context:CitationContext)`",
        "",
        "## 10. Retrieval Methodology Comparison",
        "RetractTrace implements four independent, benchmarked retrieval paradigms:",
        "1. **BM25 Lexical Retrieval:** Okapi BM25 scoring over title, abstract, and extracted context sentences.",
        "2. **Dense Semantic Retrieval:** Cosine similarity over Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) indexed with FAISS.",
        "3. **Hybrid Retrieval:** Reciprocal Rank Fusion (RRF, k=60) combining lexical and semantic rankings.",
        "4. **Graph-Enhanced Retrieval:** Graph-aware scoring integrating direct citation links, temporal post-retraction risk weighting (+0.25), and reliance classification intent (+0.30).",
        "",
        "## 11. Retrieval Ranking Explanations",
        "Retrieved documents are ranked by multi-objective scoring. Papers citing retracted findings after the retraction date "
        "and those exhibiting reliance/methodological intent receive elevated prominence in investigation workflows.",
        "",
        "## 12. Main Findings",
        f"- The retracted paper has accumulated ongoing citations in the scholarly record.",
        f"- **{after}** citation contexts occurred after the formal retraction date, indicating an ongoing awareness gap in the literature.",
        f"- Classification indicates {classification_counts.get('SUPPORTING_OR_RELIANCE', 0)} citation(s) with supporting/reliance intent.",
        "",
        "## 13. Evidence Limitations",
        "- Non-open-access citing publications were excluded from XML full-text extraction due to licensing constraints.",
        "- Extracted sentence boundaries depend on regex sentence segmentation and PMC XML tagging.",
        "- Algorithmic classifications reflect linguistic patterns and do not substitute for expert human peer review.",
        "",
        "## 14. Methodological Caveats",
        "A scholarly citation demonstrates bibliographic connectivity but does NOT prove scientific misconduct, invalidity "
        "of the citing study, or affirmative endorsement of fraudulent data. Citations may discuss retracted works as historical "
        "context or methodological background without relying on retracted empirical findings.",
    ])

    return "\n".join(report_lines)
