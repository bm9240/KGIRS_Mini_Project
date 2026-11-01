"""FastAPI endpoints backed by the local Phase 1 dataset."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from src.explainability.service import ExplanationService
from src.classification.engine import CitationClassifier, load_annotation_file, save_human_annotation
from src.evaluation.metrics import (
    classification_metrics,
    explainability_assessment,
    run_retrieval_benchmark,
    system_performance_summary,
)
from src.knowledge_graph.service import GraphService
from src.utils.config import settings
from src.preprocessing.dataset import dataset_statistics, load_dataset, read_csv
from src.retrieval.engine import (
    BM25Retriever,
    GraphEnhancedRetriever,
    reciprocal_rank_fusion,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    method: str = Field(default="bm25", pattern="^(bm25|dense|hybrid|graph|graph_enhanced)$")
    top_k: int = Field(default=10, ge=1, le=100)


class ClassificationRequest(BaseModel):
    citation_sentence: str = Field(default="", max_length=10000)
    paragraph_text: str = Field(default="", max_length=30000)
    method: str = Field(default="rule_based", pattern="^rule_based$")


class AnnotationRequest(BaseModel):
    annotation_id: str
    human_label: str
    annotator: str = "expert_reviewer"
    notes: str = ""


@lru_cache
def resources():
    dataset = load_dataset()
    bm25 = BM25Retriever(dataset)
    explanations = ExplanationService(dataset)
    graph_service = GraphService(settings, dataset)
    return dataset, bm25, explanations, graph_service


@lru_cache
def dense_resource():
    from src.retrieval.engine import DenseRetriever

    dataset, _, _, _ = resources()
    return DenseRetriever(dataset, settings.dense_model)


@lru_cache
def graph_retriever_resource():
    dataset, bm25, _, _ = resources()
    dense = None
    try:
        dense = dense_resource()
    except Exception:
        pass
    return GraphEnhancedRetriever(dataset, bm25, dense)


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:
        raise RuntimeError("Install fastapi and uvicorn to run the API.") from error

    app = FastAPI(title="RetractTrace API", version="2.0", description="Explainable Information Retrieval for Retracted Scientific Literature")

    @app.get("/health")
    def health():
        dataset, _, _, graph_service = resources()
        return {
            "status": "ok",
            "dataset": dataset_statistics(dataset),
            "neo4j_connected": graph_service.is_connected(),
        }

    @app.get("/stats")
    def stats():
        dataset, _, _, graph_service = resources()
        return system_performance_summary(dataset, graph_service)

    @app.post("/search")
    def search(request: SearchRequest):
        dataset, bm25, _, graph_service = resources()
        method_norm = "graph_enhanced" if request.method in {"graph", "graph_enhanced"} else request.method

        if method_norm == "bm25":
            results = bm25.search(request.query, request.top_k)
        elif method_norm == "dense":
            try:
                results = dense_resource().search(request.query, request.top_k)
            except (ImportError, RuntimeError) as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
        elif method_norm == "hybrid":
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
        else:
            # Genuine Graph-Enhanced retrieval
            graph_retriever = graph_retriever_resource()
            results = graph_retriever.search(request.query, request.top_k)

        return {
            "query": request.query,
            "method": method_norm,
            "results_count": len(results),
            "results": [result.__dict__ for result in results],
        }

    @app.post("/retrieval/compare")
    def compare(request: SearchRequest):
        dataset, bm25, _, _ = resources()
        dense = None
        try:
            dense = dense_resource()
        except (ImportError, RuntimeError):
            pass

        bm25_res = bm25.search(request.query, request.top_k)
        response: dict[str, Any] = {
            "query": request.query,
            "bm25": [r.__dict__ for r in bm25_res],
        }

        if dense is not None:
            dense_res = dense.search(request.query, request.top_k)
            response["dense"] = [r.__dict__ for r in dense_res]
            response["hybrid"] = [
                r.__dict__
                for r in reciprocal_rank_fusion([bm25_res, dense_res], request.top_k)
            ]
        else:
            response["dense"] = []
            response["hybrid"] = []

        graph_retriever = graph_retriever_resource()
        graph_res = graph_retriever.search(request.query, request.top_k)
        response["graph_enhanced"] = [r.__dict__ for r in graph_res]
        response["dataset_size"] = len(dataset.citing)
        return response

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
                    row.get("retracted_record_id", ""),
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
                or row.get("retracted_record_id") == paper_id
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
            "sentence_evidence": list(prediction.sentence_evidence),
            "paragraph_evidence": list(prediction.paragraph_evidence),
        }

    @app.get("/graph/subgraph/{paper_id:path}")
    def graph_subgraph(paper_id: str, max_citing: int = 35):
        _, _, _, graph_service = resources()
        return graph_service.get_subgraph_for_paper(paper_id, max_citing=max_citing)

    @app.get("/graph/paths/{paper_id:path}")
    def graph_paths(paper_id: str, hops: int = 2):
        _, _, _, graph_service = resources()
        return {"paths": graph_service.paths(paper_id, hops)}

    @app.get("/evaluation/retrieval")
    def eval_retrieval(k: int = 5):
        dataset, bm25, _, _ = resources()
        dense = None
        try:
            dense = dense_resource()
        except Exception:
            pass
        graph_retriever = graph_retriever_resource()
        return run_retrieval_benchmark(dataset, bm25, dense, reciprocal_rank_fusion, graph_retriever, k=k)

    @app.get("/evaluation/explainability")
    def eval_explainability():
        dataset = resources()[0]
        return explainability_assessment(dataset)

    @app.get("/evaluation/classification")
    def eval_classification():
        annot_path = Path(settings.root) / "data" / "annotations" / "citation_context_annotations.csv"
        rows = load_annotation_file(annot_path)
        return classification_metrics(rows)

    @app.get("/annotations")
    def list_annotations():
        annot_path = Path(settings.root) / "data" / "annotations" / "citation_context_annotations.csv"
        return load_annotation_file(annot_path)

    @app.post("/annotations/save")
    def save_annotation(req: AnnotationRequest):
        annot_path = Path(settings.root) / "data" / "annotations" / "citation_context_annotations.csv"
        success = save_human_annotation(
            annot_path,
            req.annotation_id,
            req.human_label,
            req.annotator,
            req.notes,
        )
        if not success:
            raise HTTPException(status_code=400, detail="Could not save annotation.")
        return {"status": "saved", "annotation_id": req.annotation_id, "human_label": req.human_label}

    return app


app = create_app()
