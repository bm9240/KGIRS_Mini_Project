"""Retrieval over genuine Phase 1 papers and citation evidence."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

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


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _documents(dataset: Dataset) -> list[dict[str, str]]:
    contexts_by_paper: dict[str, list[dict[str, str]]] = {}
    for row in dataset.contexts:
        contexts_by_paper.setdefault(row.get("citing_openalex_id", ""), []).append(row)
    docs = []
    for row in dataset.citing:
        paper_id = row.get("openalex_id", "")
        context_text = " ".join(
            f"{item.get('citation_sentence', '')} {item.get('paragraph_text', '')}"
            for item in contexts_by_paper.get(paper_id, [])
        )
        docs.append(
            {
                "id": paper_id,
                "title": row.get("citing_title", ""),
                "doi": row.get("citing_doi", ""),
                "text": f"{row.get('citing_title', '')} {row.get('citing_abstract', '')} {context_text}",
                "context_count": str(len(contexts_by_paper.get(paper_id, []))),
            }
        )
    return docs


class BM25Retriever:
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
            if score:
                scores.append((score, index, matched))
        scores.sort(reverse=True)
        return [
            SearchResult(
                self.documents[index]["id"],
                self.documents[index]["title"],
                self.documents[index]["doi"],
                score,
                "bm25",
                (f"Matched terms: {', '.join(matched)}",),
                int(self.documents[index]["context_count"]),
            )
            for score, index, matched in scores[:top_k]
        ]


class DenseRetriever:
    """Sentence-Transformer + FAISS retrieval; unavailable means explicit error."""

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
                self.documents[index]["id"],
                self.documents[index]["title"],
                self.documents[index]["doi"],
                float(score),
                "dense",
                ("Semantic embedding similarity.",),
                int(self.documents[index]["context_count"]),
            )
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]


def reciprocal_rank_fusion(result_sets: Iterable[list[SearchResult]], top_k: int) -> list[SearchResult]:
    combined: dict[str, tuple[float, SearchResult]] = {}
    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            score, original = combined.get(result.paper_id, (0.0, result))
            combined[result.paper_id] = (score + 1 / (60 + rank), original)
    ranked = sorted(combined.values(), key=lambda item: item[0], reverse=True)
    return [
        SearchResult(
            result.paper_id, result.title, result.doi, score, "hybrid",
            result.explanation + ("Reciprocal Rank Fusion of available retrievers.",),
            result.context_count,
        )
        for score, result in ranked[:top_k]
    ]
