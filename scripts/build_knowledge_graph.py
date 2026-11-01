"""Load the Phase 1 dataset into the configured Neo4j Aura database."""

from src.knowledge_graph.service import GraphService
from src.preprocessing.dataset import load_dataset


if __name__ == "__main__":
    graph = GraphService()
    try:
        graph.connect()
        counts = graph.load_dataset(load_dataset())
        print(f"Neo4j Aura graph load complete: {counts}")
        print(f"Node labels: {graph.label_counts()}")
    except Exception as error:
        print(f"Neo4j Aura graph load failed: {error}")
        raise SystemExit(1)
    finally:
        graph.close()
