"""Safe, rerunnable Neo4j Aura graph construction and queries."""

from __future__ import annotations

from typing import Any

from src.preprocessing.dataset import Dataset
from src.utils.config import AppSettings, settings


SCHEMA_QUERIES = (
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (n:Paper) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (n:Author) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT journal_id IF NOT EXISTS FOR (n:Journal) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT retraction_id IF NOT EXISTS FOR (n:Retraction) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT context_id IF NOT EXISTS FOR (n:CitationContext) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX paper_doi IF NOT EXISTS FOR (n:Paper) ON (n.doi)",
)


class GraphService:
    def __init__(self, config: AppSettings = settings) -> None:
        self.config = config
        self.driver: Any | None = None

    def connect(self) -> None:
        if not all(
            (self.config.neo4j_uri, self.config.neo4j_username, self.config.neo4j_password)
        ):
            raise RuntimeError("Neo4j Aura configuration is incomplete.")
        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise RuntimeError("Install neo4j to use graph operations.") from error
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
        )
        self.driver.verify_connectivity()

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j is not connected.")
        with self.driver.session(database=self.config.neo4j_database) as session:
            return [record.data() for record in session.run(query, **parameters)]

    def initialize_schema(self) -> None:
        for query in SCHEMA_QUERIES:
            self._run(query)

    def load_dataset(self, dataset: Dataset) -> dict[str, int]:
        self.initialize_schema()
        self._run("MATCH (c:CitationContext) DETACH DELETE c")
        for row in dataset.retracted:
            paper_id = row.get("openalex_id") or row.get("retracted_doi") or row.get("retracted_pmid")
            self._run(
                """
                MERGE (p:Paper {id: $id})
                SET p.title=$title, p.doi=$doi, p.pmid=$pmid,
                    p.publication_date=$publication_date, p.source='retraction_watch',
                    p.retraction_reasons=$reasons
                MERGE (r:Retraction {id: $id + ':retraction'})
                SET r.date=$retraction_date, r.reasons=$reasons
                MERGE (p)-[:RETRACTED_FOR]->(r)
                """,
                id=paper_id,
                title=row.get("retracted_title", ""),
                doi=row.get("retracted_doi", ""),
                pmid=row.get("retracted_pmid", ""),
                publication_date=row.get("original_paper_date", ""),
                retraction_date=row.get("retraction_date", ""),
                reasons=row.get("retraction_reasons", ""),
            )
        for row in dataset.citing:
            paper_id = row["openalex_id"]
            self._run(
                """
                MERGE (p:Paper {id: $id})
                SET p.title=$title, p.doi=$doi, p.pmid=$pmid,
                    p.publication_date=$date, p.pmcid=$pmcid,
                    p.source='openalex', p.full_text_status=$xml_status
                """,
                id=paper_id, title=row.get("citing_title", ""),
                doi=row.get("citing_doi", ""), pmid=row.get("citing_pmid", ""),
                date=row.get("publication_date", ""), pmcid=row.get("pmcid", ""),
                xml_status=row.get("xml_status", ""),
            )
        for occurrence, row in enumerate(dataset.contexts, start=1):
            context_id = row.get("context_id") or f"occurrence:{occurrence}"
            self._run(
                """
                MERGE (c: CitationContext {id: $id})
                SET c.sentence=$sentence, c.paragraph=$paragraph,
                    c.section=$section, c.timing=$timing,
                    c.matching_method=$matching_method,
                    c.matching_confidence=$matching_confidence,
                    c.source=$xml_file
                WITH c
                MATCH (citing:Paper {id: $citing_id})
                MATCH (retracted:Paper {doi: $retracted_doi})
                MERGE (citing)-[:CITES]->(retracted)
                MERGE (citing)-[:HAS_CONTEXT]->(c)
                """,
                id=context_id, sentence=row.get("citation_sentence", ""),
                paragraph=row.get("paragraph_text", ""), section=row.get("section_heading", ""),
                timing=row.get("citation_timing", ""), matching_method=row.get("matching_method", ""),
                matching_confidence=row.get("matching_confidence", ""),
                xml_file=row.get("xml_file", ""), citing_id=row["citing_openalex_id"],
                retracted_doi=row.get("retracted_doi", ""),
            )
        return self.counts()

    def counts(self) -> dict[str, int]:
        node_rows = self._run("MATCH (n) RETURN count(n) AS nodes")
        relationship_rows = self._run("MATCH ()-[r]->() RETURN count(r) AS relationships")
        return {
            "nodes": node_rows[0]["nodes"] if node_rows else 0,
            "relationships": (
                relationship_rows[0]["relationships"] if relationship_rows else 0
            ),
        }

    def label_counts(self) -> list[dict[str, Any]]:
        return self._run(
            "MATCH (n) RETURN labels(n) AS labels, count(n) AS count ORDER BY labels"
        )

    def search_papers(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return self._run(
            """
            MATCH (p:Paper)
            WHERE toLower(coalesce(p.title,'')) CONTAINS toLower($query_text)
               OR toLower(coalesce(p.doi,'')) CONTAINS toLower($query_text)
            RETURN p.id AS id, p.title AS title, p.doi AS doi,
                   p.publication_date AS publication_date
            ORDER BY p.title LIMIT $limit
            """,
            query_text=query,
            limit=limit,
        )

    def paths(self, paper_id: str, hops: int = 2) -> list[dict[str, Any]]:
        hops = max(1, min(hops, 4))
        return self._run(
            f"""
            MATCH path=(start:Paper {{id: $paper_id}})-[*1..{hops}]-(end:Paper)
            RETURN [node IN nodes(path) | node.id] AS nodes,
                   [rel IN relationships(path) | type(rel)] AS relationships
            LIMIT 100
            """,
            paper_id=paper_id,
        )
