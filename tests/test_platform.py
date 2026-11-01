"""Comprehensive regression and research validation test suite for RetractTrace."""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app
from src.classification.engine import CATEGORIES, CitationClassifier, RuleBasedClassifier
from src.evaluation.metrics import (
    explainability_assessment,
    ndcg_at_k,
    retrieval_metrics,
    run_retrieval_benchmark,
)
from src.explainability.service import ExplanationService, build_investigation_report_markdown
from src.knowledge_graph.service import GraphService
from src.preprocessing.dataset import dataset_statistics, load_dataset
from src.retrieval.engine import (
    BM25Retriever,
    DenseRetriever,
    GraphEnhancedRetriever,
    reciprocal_rank_fusion,
)
from src.utils.config import settings


class RetractTracePlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()
        cls.bm25 = BM25Retriever(cls.dataset)
        cls.dense = DenseRetriever(cls.dataset, settings.dense_model)
        cls.graph_retriever = GraphEnhancedRetriever(cls.dataset, cls.bm25, cls.dense)
        cls.graph_service = GraphService(settings, cls.dataset)
        cls.client = TestClient(app)
        cls.explainer = ExplanationService(cls.dataset)

    # 1. Dataset Integrity
    def test_dataset_integrity_counts(self):
        stats = dataset_statistics(self.dataset)
        self.assertEqual(stats["retracted_papers"], 100)
        self.assertEqual(stats["citing_papers"], 915)
        self.assertEqual(stats["citation_contexts"], 148)
        self.assertEqual(stats["contexts_before_retraction"], 94)
        self.assertEqual(stats["contexts_after_retraction"], 54)
        self.assertEqual(stats["xml_downloaded"], 104)

    # 2. Classification Outputs & Rules
    def test_classification_categories_and_rules(self):
        classifier = RuleBasedClassifier()

        # Retraction acknowledgement
        p_ret = classifier.predict("This article was retracted due to image errors.")
        self.assertEqual(p_ret.category, "RETRACTION_ACKNOWLEDGEMENT")
        self.assertGreaterEqual(p_ret.confidence, 0.90)

        # Critical
        p_crit = classifier.predict("Our findings contradict and refute the previous claims.")
        self.assertEqual(p_crit.category, "CRITICAL")

        # Methodological
        p_meth = classifier.predict("The nanoparticles were synthesized according to the protocol described.")
        self.assertEqual(p_meth.category, "METHODOLOGICAL")

        # Supporting
        p_supp = classifier.predict("Our experimental results are consistent with and confirm the findings.")
        self.assertEqual(p_supp.category, "SUPPORTING_OR_RELIANCE")

        # Background
        p_bg = classifier.predict("Previous studies in the literature have reviewed this mechanism.")
        self.assertEqual(p_bg.category, "BACKGROUND")

        # Unclear / Neutral
        p_unc = classifier.predict("A study was referenced.")
        self.assertEqual(p_unc.category, "UNCLEAR")

    # 3. Temporal Analysis & Explanations
    def test_temporal_classification(self):
        ctx_after = next(c for c in self.dataset.contexts if c.get("citation_timing") == "AFTER_RETRACTION")
        exp = self.explainer.context_explanation(ctx_after)
        self.assertEqual(exp["temporal_analysis"]["timing"], "AFTER_RETRACTION")
        self.assertIsNotNone(exp["retracted_paper"]["title"])
        self.assertIsNotNone(exp["citing_paper"]["title"])

        ctx_before = next(c for c in self.dataset.contexts if c.get("citation_timing") == "BEFORE_RETRACTION")
        exp_b = self.explainer.context_explanation(ctx_before)
        self.assertEqual(exp_b["temporal_analysis"]["timing"], "BEFORE_RETRACTION")

    # 4. Missing Citation Contexts Graceful Handling
    def test_missing_citation_contexts_handling(self):
        # A citing paper without extracted XML
        no_xml = next(c for c in self.dataset.citing if c.get("xml_status") != "downloaded")
        exp = self.explainer.explain_paper(no_xml["openalex_id"])
        self.assertEqual(exp["evidence_status"], "no_extracted_contexts")
        self.assertEqual(exp["contexts_count"], 0)

    # 5. All Four Retrieval Approaches
    def test_all_four_retrieval_methods_execute_independently(self):
        q = "ovarian cancer drug resistance cold responsive nanomaterial"
        k = 3

        bm25_res = self.bm25.search(q, top_k=k)
        self.assertLessEqual(len(bm25_res), k)
        self.assertTrue(all(r.method == "bm25" for r in bm25_res))

        dense_res = self.dense.search(q, top_k=k)
        self.assertLessEqual(len(dense_res), k)
        self.assertTrue(all(r.method == "dense" for r in dense_res))

        hybrid_res = reciprocal_rank_fusion([bm25_res, dense_res], top_k=k)
        self.assertLessEqual(len(hybrid_res), k)
        self.assertTrue(all(r.method == "hybrid" for r in hybrid_res))

        graph_res = self.graph_retriever.search(q, top_k=k)
        self.assertLessEqual(len(graph_res), k)
        self.assertTrue(all(r.method == "graph_enhanced" for r in graph_res))
        self.assertTrue(all("composite_score" in r.ranking_factors for r in graph_res))

    # 6. Graph Path and Subgraph Correctness
    def test_graph_subgraph_and_paths(self):
        subgraph = self.graph_service.get_subgraph_for_paper("70462", max_citing=20)
        self.assertIn("nodes", subgraph)
        self.assertIn("edges", subgraph)
        self.assertGreater(len(subgraph["nodes"]), 0)
        self.assertGreater(len(subgraph["edges"]), 0)

        # Retracted paper is central node
        ret_node = next(n for n in subgraph["nodes"] if n["node_type"] == "retracted")
        self.assertEqual(ret_node["title"], "Overcoming Ovarian Cancer Drug Resistance with a Cold Responsive Nanomaterial")

        # Directed CITES edges point from Citing -> Retracted
        cites_edge = next(e for e in subgraph["edges"] if e["type"] == "CITES")
        self.assertEqual(cites_edge["target"], ret_node["id"])

        paths = self.graph_service.paths("W2800209693", hops=2)
        self.assertGreater(len(paths), 0)

    # 7. FastAPI Endpoints
    def test_fastapi_endpoints(self):
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)

        res_stats = self.client.get("/stats")
        self.assertEqual(res_stats.status_code, 200)

        res_search = self.client.post("/search", json={"query": "pancreatic cancer", "method": "bm25", "top_k": 3})
        self.assertEqual(res_search.status_code, 200)
        self.assertIn("results", res_search.json())

        res_compare = self.client.post("/retrieval/compare", json={"query": "pancreatic cancer", "top_k": 3})
        self.assertEqual(res_compare.status_code, 200)
        data = res_compare.json()
        self.assertIn("bm25", data)
        self.assertIn("dense", data)
        self.assertIn("hybrid", data)
        self.assertIn("graph_enhanced", data)

        res_audit = self.client.get("/evaluation/explainability")
        self.assertEqual(res_audit.status_code, 200)

    # 8. Report Generation
    def test_report_generation_all_14_sections(self):
        record = next(r for r in self.dataset.retracted if r["retracted_record_id"] == "70462")
        contexts = [c for c in self.dataset.contexts if c.get("retracted_record_id") == "70462"]
        report = build_investigation_report_markdown(record, contexts)

        for section_num in range(1, 15):
            self.assertIn(f"## {section_num}.", report)

    # 9. Case 1 Acceptance: Ovarian Cancer
    def test_acceptance_case_1_ovarian_cancer(self):
        rec_id = "70462"
        doi = "10.1021/acscentsci.8b00050"
        retracted = next(r for r in self.dataset.retracted if r["retracted_record_id"] == rec_id)
        self.assertEqual(retracted["retracted_doi"], doi)

        # Citing publications verification
        citing = [c for c in self.dataset.citing if rec_id in c["retracted_record_ids"].split(";")]
        citing_titles = [c["citing_title"].lower() for c in citing]
        self.assertTrue(any("nanoparticle-based drug delivery in cancer therapy" in t for t in citing_titles))
        self.assertTrue(any("recent progress in cryoablation cancer therapy" in t for t in citing_titles))
        self.assertTrue(any("deciphering resistance mechanisms" in t for t in citing_titles))

        # Contexts exist
        ctxs = [c for c in self.dataset.contexts if c.get("retracted_record_id") == rec_id]
        self.assertGreater(len(ctxs), 0)
        timings = {c["citation_timing"] for c in ctxs}
        self.assertIn("BEFORE_RETRACTION", timings)
        self.assertIn("AFTER_RETRACTION", timings)

    # 10. Case 2 Acceptance: Pancreatic Cancer
    def test_acceptance_case_2_pancreatic_cancer(self):
        rec_id = "66404"
        doi = "10.1093/abbs/gmz096"
        retracted = next(r for r in self.dataset.retracted if r["retracted_record_id"] == rec_id)
        self.assertEqual(retracted["retracted_doi"], doi)
        self.assertIn("miR-129-5p", retracted["retracted_title"])

        ctxs = [c for c in self.dataset.contexts if c.get("retracted_record_id") == rec_id]
        self.assertGreater(len(ctxs), 0)
        timings = {c["citation_timing"] for c in ctxs}
        self.assertIn("BEFORE_RETRACTION", timings)
        self.assertIn("AFTER_RETRACTION", timings)

    # 11. Case 3 Acceptance: Materials Science
    def test_acceptance_case_3_materials_science(self):
        rec_id = "53194"
        doi = "10.1021/acsami.9b23052"
        retracted = next(r for r in self.dataset.retracted if r["retracted_record_id"] == rec_id)
        self.assertEqual(retracted["retracted_doi"], doi)
        self.assertIn("Cellulose/rGO/Fe3O4", retracted["retracted_title"])

        ctxs = [c for c in self.dataset.contexts if c.get("retracted_record_id") == rec_id]
        self.assertGreater(len(ctxs), 0)
        timings = {c["citation_timing"] for c in ctxs}
        self.assertIn("BEFORE_RETRACTION", timings)
        self.assertIn("AFTER_RETRACTION", timings)


if __name__ == "__main__":
    unittest.main()
