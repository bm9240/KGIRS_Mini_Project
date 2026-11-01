"""FastAPI endpoints backed by the local Phase 1 dataset."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from src.explainability.service import ExplanationService
from src.classification.engine import CitationClassifier
from src.evaluation.metrics import classification_metrics
from src.knowledge_graph.service import GraphService
from src.utils.config import settings
from src.preprocessing.dataset import dataset_statistics, load_dataset
from src.retrieval.engine import BM25Retriever, reciprocal_rank_fusion


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    method: str = Field(default="bm25", pattern="^(bm25|dense|hybrid|graph)$")
    top_k: int = Field(default=10, ge=1, le=100)


class ClassificationRequest(BaseModel):
    citation_sentence: str = Field(default="", max_length=10000)
    paragraph_text: str = Field(default="", max_length=30000)
    method: str = Field(default="rule_based", pattern="^rule_based$")


@lru_cache
def resources():
    dataset = load_dataset()
    return dataset, BM25Retriever(dataset), ExplanationService(dataset)


@lru_cache
def dense_resource():
    from src.retrieval.engine import DenseRetriever

    dataset, _, _ = resources()
    return DenseRetriever(dataset, settings.dense_model)


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:
        raise RuntimeError("Install fastapi and uvicorn to run the API.") from error

    app = FastAPI(title="RetractTrace API", version="1.0")

    @app.get("/health")
    def health():
        dataset, _, _ = resources()
        return {"status": "ok", "dataset": dataset_statistics(dataset)}

    @app.get("/stats")
    def stats():
        return dataset_statistics(resources()[0])

    @app.post("/search")
    def search(request: SearchRequest):
        dataset, bm25, _ = resources()
        if request.method == "bm25":
            results = bm25.search(request.query, request.top_k)
        elif request.method == "hybrid":
            try:
                dense = dense_resource()
            except (ImportError, RuntimeError) as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            results = reciprocal_rank_fusion(
                [
                    bm25.search(request.query, request.top_k),
                    dense.search(request.query, request.top_k),
                ],
                request.top_k,
            )
        elif request.method == "dense":
            try:
                results = dense_resource().search(request.query, request.top_k)
            except (ImportError, RuntimeError) as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
        else:
            graph = GraphService()
            try:
                graph.connect()
                graph_rows = graph.search_papers(request.query, request.top_k)
            except RuntimeError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            finally:
                graph.close()
            results = [
                {
                    "paper": row,
                    "score": 1.0 / rank,
                    "method": "graph",
                    "explanation": ["Matched a Paper node in Neo4j Aura."],
                }
                for rank, row in enumerate(graph_rows, start=1)
            ]
        return {
            "method": request.method,
            "results": [
                result if isinstance(result, dict) else result.__dict__
                for result in results
            ],
        }

    @app.get("/papers/{paper_id}/explanation")
    def explanation(paper_id: str):
        return resources()[2].explain_paper(paper_id)

    @app.get("/contexts")
    def contexts(limit: int = 50):
        return resources()[0].contexts[: max(1, min(limit, 500))]

    @app.get("/contexts/{context_id}")
    def context(context_id: int):
        contexts = resources()[0].contexts
        if context_id < 1 or context_id > len(contexts):
            raise HTTPException(status_code=404, detail="Citation context not found.")
        return resources()[2].context_explanation(contexts[context_id - 1])

    @app.get("/papers/{paper_id:path}")
    def paper(paper_id: str):
        dataset = resources()[0]
        citing = next(
            (row for row in dataset.citing if row.get("openalex_id") == paper_id),
            None,
        )
        retracted = next(
            (
                row for row in dataset.retracted
                if paper_id in {
                    row.get("openalex_id", ""),
                    row.get("retracted_doi", ""),
                    row.get("retracted_pmid", ""),
                }
            ),
            None,
        )
        if citing is None and retracted is None:
            raise HTTPException(status_code=404, detail="Paper not found.")
        return {
            "citing": citing,
            "retracted": retracted,
            "contexts": [
                row for row in dataset.contexts
                if row.get("citing_openalex_id") == paper_id
            ],
        }

    @app.post("/classification/predict")
    def classify(request: ClassificationRequest):
        prediction = CitationClassifier().predict(
            {
                "citation_sentence": request.citation_sentence,
                "paragraph_text": request.paragraph_text,
            },
            request.method,
        )
        return {
            "category": prediction.category,
            "confidence": prediction.confidence,
            "evidence": list(prediction.evidence),
            "method": prediction.method,
        }

    @app.post("/retrieval/compare")
    def compare(request: SearchRequest):
        dataset, bm25, _ = resources()
        dense = None
        try:
            dense = dense_resource()
        except (ImportError, RuntimeError):
            pass
        response: dict[str, object] = {
            "query": request.query,
            "bm25": [result.__dict__ for result in bm25.search(request.query, request.top_k)],
        }
        if dense is not None:
            dense_results = dense.search(request.query, request.top_k)
            response["dense"] = [result.__dict__ for result in dense_results]
            response["hybrid"] = [
                result.__dict__
                for result in reciprocal_rank_fusion(
                    [bm25.search(request.query, request.top_k), dense_results],
                    request.top_k,
                )
            ]
        else:
            response["dense"] = []
            response["hybrid"] = []
        response["dataset_size"] = len(dataset.citing)
        return response

    @app.get("/evaluation")
    def evaluation():
        path = Path(settings.root) / "data" / "processed" / "classified_contexts.csv"
        if not path.is_file():
            return {"status": "not_available", "reason": "Classification output is missing."}
        from src.preprocessing.dataset import read_csv

        return classification_metrics(read_csv(path))

    @app.get("/data/status")
    def data_status():
        return dataset_statistics(resources()[0])

    @app.get("/graph/paths/{paper_id:path}")
    def graph_paths(paper_id: str, hops: int = 2):
        graph = GraphService()
        try:
            graph.connect()
            return {"paths": graph.paths(paper_id, hops)}
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        finally:
            graph.close()

    return app


app = create_app()
