"""Focused tests for the data-backed platform core."""

import unittest

from src.classification.engine import RuleBasedClassifier
from src.preprocessing.dataset import dataset_statistics, load_dataset
from src.retrieval.engine import BM25Retriever


class PlatformCoreTests(unittest.TestCase):
    def test_phase_one_dataset_is_loaded_without_empty_core_outputs(self):
        dataset = load_dataset()
        stats = dataset_statistics(dataset)
        self.assertEqual(stats["retracted_papers"], 100)
        self.assertEqual(stats["citing_papers"], 915)
        self.assertEqual(stats["citation_contexts"], 148)

    def test_bm25_returns_genuine_dataset_records(self):
        results = BM25Retriever(load_dataset()).search("retracted paper", top_k=3)
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all(result.paper_id for result in results))

    def test_rule_classifier_preserves_uncertainty(self):
        prediction = RuleBasedClassifier().predict("A paper was mentioned.")
        self.assertEqual(prediction.category, "UNCLEAR")


if __name__ == "__main__":
    unittest.main()
