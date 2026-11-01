"""Retrieval over genuine Phase 1 papers and citation evidence."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.classification.engine import CitationClassifier
from src.preprocessing.dataset import Dataset


@dataclass(frozen=True)
class SearchResult:
    paper_id: str
    title: str
    doi: str
    score: float
    method: str
    explanation: tuple[str, ...]
    context_count: int
    citation_timing: str = "NO_EXTRACTED_CONTEXT"
    citation_intent: str = "UNCLEAR"
    graph_evidence: tuple[str, ...] = ()
    ranking_factors: dict[str, float] = field(default_factory=dict)
    citation_sentence: str = ""
    paragraph_text: str = ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _documents(dataset: Dataset) -> list[dict[str, Any]]:
    contexts_by_paper: dict[str, list[dict[str, str]]] = {}
    for row in dataset.contexts:
        contexts_by_paper.setdefault(row.get("citing_openalex_id", ""), []).append(row)

    classifier = CitationClassifier()
    docs = []
    for row in dataset.citing:
        paper_id = row.get("openalex_id", "")
        contexts = contexts_by_paper.get(paper_id, [])
        context_text = " ".join(
            f"{item.get('citation_sentence', '')} {item.get('paragraph_text', '')}"
            for item in contexts
        )

        timings = [item.get("citation_timing") for item in contexts if item.get("citation_timing")]
        primary_timing = "AFTER_RETRACTION" if "AFTER_RETRACTION" in timings else ("BEFORE_RETRACTION" if "BEFORE_RETRACTION" in timings else "NO_EXTRACTED_CONTEXT")

        primary_sentence = contexts[0].get("citation_sentence", "") if contexts else ""
        primary_paragraph = contexts[0].get("paragraph_text", "") if contexts else ""

        intent = "UNCLEAR"
        if contexts:
            pred = classifier.predict(contexts[0])
            intent = pred.category

        docs.append(
            {
                "id": paper_id,
                "title": row.get("citing_title", ""),
                "doi": row.get("citing_doi", ""),
                "retracted_record_ids": [r for r in row.get("retracted_record_ids", "").split(";") if r],
                "text": f"{row.get('citing_title', '')} {row.get('citing_abstract', '')} {context_text}",
                "context_count": len(contexts),
                "contexts": contexts,
                "citation_timing": primary_timing,
                "citation_intent": intent,
                "citation_sentence": primary_sentence,
                "paragraph_text": primary_paragraph,
            }
        )
    return docs


class BM25Retriever:
    """Okapi BM25 sparse lexical retriever."""

    def __init__(self, dataset: Dataset) -> None:
        self.documents = _documents(dataset)
        self.tokens = [_tokens(doc["text"]) for doc in self.documents]
        self.avg_length = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        self.k1 = 1.5
        self.b = 0.75
        self.document_frequency: dict[str, int] = {}
        for document in self.tokens:
            for token in set(document):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_tokens = _tokens(query)
        scores: list[tuple[float, int, list[str]]] = []
        total = len(self.documents)
        for index, document in enumerate(self.tokens):
            length = len(document)
            score = 0.0
            matched = []
            frequencies = {token: document.count(token) for token in set(query_tokens)}
            for token, frequency in frequencies.items():
                if not frequency:
                    continue
                matched.append(token)
                df = self.document_frequency.get(token, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.avg_length, 1)
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((score, index, matched))
        scores.sort(reverse=True)
        return [
            SearchResult(
                paper_id=self.documents[index]["id"],
                title=self.documents[index]["title"],
                doi=self.documents[index]["doi"],
                score=round(score, 4),
                method="bm25",
                explanation=(f"BM25 term matches: {', '.join(matched)}",),
                context_count=self.documents[index]["context_count"],
                citation_timing=self.documents[index]["citation_timing"],
                citation_intent=self.documents[index]["citation_intent"],
                citation_sentence=self.documents[index]["citation_sentence"],
                paragraph_text=self.documents[index]["paragraph_text"],
                ranking_factors={"bm25_raw_score": round(score, 4)},
            )
            for score, index, matched in scores[:top_k]
        ]


class DenseRetriever:
    """Sentence-Transformer + FAISS dense semantic retriever."""

    def __init__(self, dataset: Dataset, model_name: str) -> None:
        self.documents = _documents(dataset)
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers and faiss-cpu."
            ) from error
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            [doc["text"] for doc in self.documents],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores, indices = self.index.search(vector, top_k)
        return [
            SearchResult(
                paper_id=self.documents[index]["id"],
                title=self.documents[index]["title"],
                doi=self.documents[index]["doi"],
                score=round(float(score), 4),
                method="dense",
                explanation=("Cosine similarity over Sentence-Transformer embeddings.",),
                context_count=self.documents[index]["context_count"],
                citation_timing=self.documents[index]["citation_timing"],
                citation_intent=self.documents[index]["citation_intent"],
                citation_sentence=self.documents[index]["citation_sentence"],
                paragraph_text=self.documents[index]["paragraph_text"],
                ranking_factors={"dense_cosine_similarity": round(float(score), 4)},
            )
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]


def reciprocal_rank_fusion(result_sets: Iterable[list[SearchResult]], top_k: int) -> list[SearchResult]:
    """Combines multiple ranked lists using Reciprocal Rank Fusion (k=60)."""
    combined: dict[str, tuple[float, SearchResult, list[str]]] = {}
    for r_list in result_sets:
        for rank, result in enumerate(r_list, start=1):
            rrf_score = 1.0 / (60 + rank)
            if result.paper_id not in combined:
                combined[result.paper_id] = (
                    rrf_score,
                    result,
                    [f"{result.method} rank #{rank} (RRF +{rrf_score:.4f})"],
                )
            else:
                existing_score, orig_res, explanations = combined[result.paper_id]
                combined[result.paper_id] = (
                    existing_score + rrf_score,
                    orig_res,
                    explanations + [f"{result.method} rank #{rank} (RRF +{rrf_score:.4f})"],
                )
    ranked = sorted(combined.values(), key=lambda item: item[0], reverse=True)
    return [
        SearchResult(
            paper_id=result.paper_id,
            title=result.title,
            doi=result.doi,
            score=round(score, 4),
            method="hybrid",
            explanation=tuple(explanations) + ("Reciprocal Rank Fusion (BM25 + Dense).",),
            context_count=result.context_count,
            citation_timing=result.citation_timing,
            citation_intent=result.citation_intent,
            citation_sentence=result.citation_sentence,
            paragraph_text=result.paragraph_text,
            ranking_factors={"rrf_composite_score": round(score, 4)},
        )
        for score, result, explanations in ranked[:top_k]
    ]


class GraphEnhancedRetriever:
    """Scientifically grounded, transparent Graph-Enhanced Retrieval.

    Combines semantic relevance with scholarly knowledge graph topology:
    - Direct citation links to retracted publications
    - Citation timing (post-retraction citations receive heightened risk weighting)
    - Citation intent classification (e.g. active reliance vs critique)
    - Full-text citation context availability and graph path centrality.
    """

    def __init__(
        self,
        dataset: Dataset,
        bm25: BM25Retriever,
        dense: DenseRetriever | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.dataset = dataset
        self.bm25 = bm25
        self.dense = dense
        self.documents_by_id = {doc["id"]: doc for doc in bm25.documents}
        self.retracted_by_id = {
            r.get("retracted_record_id", ""): r for r in dataset.retracted
        }
        self.weights = weights or {
            "semantic": 0.40,
            "lexical": 0.20,
            "graph_link": 0.15,
            "temporal_risk": 0.15,
            "intent_reliance": 0.10,
        }

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        candidate_pool: dict[str, dict[str, Any]] = {}

        # 1. Lexical candidate gathering
        bm25_results = self.bm25.search(query, top_k=max(30, top_k * 3))
        max_bm25 = max((r.score for r in bm25_results), default=1.0)
        for r in bm25_results:
            candidate_pool[r.paper_id] = {
                "bm25_norm": (r.score / max_bm25) if max_bm25 > 0 else 0.0,
                "dense_norm": 0.0,
            }

        # 2. Dense semantic candidate gathering
        if self.dense is not None:
            dense_results = self.dense.search(query, top_k=max(30, top_k * 3))
            max_dense = max((r.score for r in dense_results), default=1.0)
            for r in dense_results:
                if r.paper_id not in candidate_pool:
                    candidate_pool[r.paper_id] = {"bm25_norm": 0.0, "dense_norm": 0.0}
                candidate_pool[r.paper_id]["dense_norm"] = max(0.0, r.score / max_dense if max_dense > 0 else 0.0)

        scored_results: list[SearchResult] = []

        for paper_id, base_scores in candidate_pool.items():
            doc = self.documents_by_id.get(paper_id)
            if not doc:
                continue

            lexical_score = base_scores["bm25_norm"]
            semantic_score = base_scores["dense_norm"]

            # Graph topology calculations
            retracted_ids = doc["retracted_record_ids"]
            has_direct_retraction_link = bool(retracted_ids)
            context_count = doc["context_count"]
            citation_timing = doc["citation_timing"]
            citation_intent = doc["citation_intent"]

            # Graph link factor (0 to 1)
            # Having confirmed citation link to retracted entity: 0.70; with verified contexts: 1.0
            graph_link_score = 0.0
            if has_direct_retraction_link:
                graph_link_score = 0.70 if context_count == 0 else 1.0

            # Temporal risk factor
            # Citations occurring AFTER retraction present higher epistemic contamination risk!
            temporal_score = 0.0
            temporal_label = "Date uncertain or no context"
            if citation_timing == "AFTER_RETRACTION":
                temporal_score = 1.0  # High risk: Continued post-retraction citation
                temporal_label = "Cited AFTER retraction (epistemic risk +1.0)"
            elif citation_timing == "BEFORE_RETRACTION":
                temporal_score = 0.40  # Pre-retraction citation
                temporal_label = "Cited before retraction (+0.40)"

            # Intent / Reliance factor
            # Supporting/reliance indicates direct scientific dependence
            intent_score = 0.10
            intent_label = f"Intent: {citation_intent}"
            if citation_intent == "SUPPORTING_OR_RELIANCE":
                intent_score = 1.0
                intent_label = "Supporting/reliance on retracted findings (+1.0)"
            elif citation_intent == "METHODOLOGICAL":
                intent_score = 0.70
                intent_label = "Methodological adoption from retracted work (+0.70)"
            elif citation_intent == "CRITICAL":
                intent_score = 0.50
                intent_label = "Critical engagement with retracted work (+0.50)"
            elif citation_intent == "BACKGROUND":
                intent_score = 0.35
                intent_label = "Background scholarly context (+0.35)"
            elif citation_intent == "RETRACTION_ACKNOWLEDGEMENT":
                intent_score = 0.20
                intent_label = "Retraction acknowledgement noted (+0.20)"

            # Composite graph-aware score
            composite = (
                self.weights["semantic"] * semantic_score
                + self.weights["lexical"] * lexical_score
                + self.weights["graph_link"] * graph_link_score
                + self.weights["temporal_risk"] * temporal_score
                + self.weights["intent_reliance"] * intent_score
            )

            # Build explainable graph evidence
            graph_evidence_items = []
            for rid in retracted_ids:
                ret_p = self.retracted_by_id.get(rid, {})
                ret_title = ret_p.get("retracted_title", f"Paper #{rid}")
                graph_evidence_items.append(f"CITES retracted paper: '{ret_title[:60]}...'")

            if context_count > 0:
                graph_evidence_items.append(f"HAS_CONTEXT: {context_count} verified XML citation context(s)")

            explanation = (
                f"Graph-Enhanced Score: {composite:.4f}",
                f"Semantic: {semantic_score:.2f} · Lexical: {lexical_score:.2f}",
                f"Graph connection: {graph_link_score:.2f} ({len(retracted_ids)} retracted target(s))",
                temporal_label,
                intent_label,
            )

            ranking_factors = {
                "composite_score": round(composite, 4),
                "semantic_component": round(self.weights["semantic"] * semantic_score, 4),
                "lexical_component": round(self.weights["lexical"] * lexical_score, 4),
                "graph_link_component": round(self.weights["graph_link"] * graph_link_score, 4),
                "temporal_risk_component": round(self.weights["temporal_risk"] * temporal_score, 4),
                "intent_reliance_component": round(self.weights["intent_reliance"] * intent_score, 4),
            }

            scored_results.append(
                SearchResult(
                    paper_id=doc["id"],
                    title=doc["title"],
                    doi=doc["doi"],
                    score=round(composite, 4),
                    method="graph_enhanced",
                    explanation=explanation,
                    context_count=context_count,
                    citation_timing=citation_timing,
                    citation_intent=citation_intent,
                    graph_evidence=tuple(graph_evidence_items),
                    ranking_factors=ranking_factors,
                    citation_sentence=doc["citation_sentence"],
                    paragraph_text=doc["paragraph_text"],
                )
            )

        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:top_k]
