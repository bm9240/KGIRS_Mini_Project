"""Safe, rerunnable Neo4j Aura and fallback local graph construction and queries."""

from __future__ import annotations

import logging
from typing import Any

from src.preprocessing.dataset import Dataset, load_dataset
from src.utils.config import AppSettings, settings

LOGGER = logging.getLogger("retracttrace.graph")

SCHEMA_QUERIES = (
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (n:Paper) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (n:Author) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT journal_id IF NOT EXISTS FOR (n:Journal) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT retraction_id IF NOT EXISTS FOR (n:Retraction) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT context_id IF NOT EXISTS FOR (n:CitationContext) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX paper_doi IF NOT EXISTS FOR (n:Paper) ON (n.doi)",
    "CREATE INDEX paper_record_id IF NOT EXISTS FOR (n:Paper) ON (n.record_id)",
)


class GraphService:
    """Scholarly Knowledge Graph service backed by Neo4j with local fallback."""

    def __init__(self, config: AppSettings = settings, dataset: Dataset | None = None) -> None:
        self.config = config
        self.driver: Any | None = None
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            self._dataset = load_dataset()
        return self._dataset

    def is_connected(self) -> bool:
        if self.driver is None:
            self.connect()
        if self.driver is None:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def connect(self) -> bool:
        if not all(
            (self.config.neo4j_uri, self.config.neo4j_username, self.config.neo4j_password)
        ):
            return False
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_username, self.config.neo4j_password),
            )
            self.driver.verify_connectivity()
            return True
        except Exception as error:
            LOGGER.warning("Could not connect to Neo4j (%s): %s", self.config.neo4j_uri, error)
            self.driver = None
            return False

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = None

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j is not connected.")
        with self.driver.session(database=self.config.neo4j_database) as session:
            return [record.data() for record in session.run(query, **parameters)]

    def initialize_schema(self) -> None:
        if not self.is_connected():
            return
        for query in SCHEMA_QUERIES:
            try:
                self._run(query)
            except Exception as e:
                LOGGER.debug("Schema query failed: %s", e)

    def load_dataset(self, dataset: Dataset) -> dict[str, int]:
        """Populate Neo4j with verified dataset records and relationships."""
        if not self.is_connected():
            raise RuntimeError("Neo4j is not connected.")

        self.initialize_schema()
        self._run("MATCH (n) DETACH DELETE n")

        # 1. Load Retracted Papers and Retraction nodes
        for row in dataset.retracted:
            paper_id = row.get("openalex_id") or row.get("retracted_doi") or row.get("retracted_pmid") or row.get("retracted_record_id")
            self._run(
                """
                MERGE (p:Paper {id: $id})
                SET p.title=$title, p.doi=$doi, p.pmid=$pmid,
                    p.record_id=$record_id,
                    p.publication_date=$publication_date, p.source='retraction_watch',
                    p.retraction_date=$retraction_date,
                    p.retraction_reasons=$reasons,
                    p.journal=$journal,
                    p.is_retracted=true
                MERGE (r:Retraction {id: $id + ':retraction'})
                SET r.date=$retraction_date, r.reasons=$reasons, r.record_id=$record_id
                MERGE (p)-[:RETRACTED_FOR]->(r)
                """,
                id=paper_id,
                record_id=row.get("retracted_record_id", ""),
                title=row.get("retracted_title", ""),
                doi=row.get("retracted_doi", ""),
                pmid=row.get("retracted_pmid", ""),
                publication_date=row.get("original_paper_date", ""),
                retraction_date=row.get("retraction_date", ""),
                reasons=row.get("retraction_reasons", ""),
                journal=row.get("journal", ""),
            )

        # 2. Load Citing Papers and direct CITES links from OpenAlex records
        for row in dataset.citing:
            paper_id = row["openalex_id"]
            self._run(
                """
                MERGE (p:Paper {id: $id})
                SET p.title=$title, p.doi=$doi, p.pmid=$pmid,
                    p.publication_date=$date, p.pmcid=$pmcid,
                    p.authors=$authors, p.journal=$journal,
                    p.source='openalex', p.full_text_status=$xml_status,
                    p.is_retracted=false
                """,
                id=paper_id, title=row.get("citing_title", ""),
                doi=row.get("citing_doi", ""), pmid=row.get("citing_pmid", ""),
                date=row.get("publication_date", ""), pmcid=row.get("pmcid", ""),
                authors=row.get("citing_authors", ""), journal=row.get("citing_journal", ""),
                xml_status=row.get("xml_status", ""),
            )
            # Create CITES relationships to all cited retracted papers
            for rid in filter(None, row.get("retracted_record_ids", "").split(";")):
                self._run(
                    """
                    MATCH (citing:Paper {id: $citing_id})
                    MATCH (retracted:Paper)
                    WHERE retracted.record_id = $rid OR retracted.doi = $rid OR retracted.id = $rid
                    MERGE (citing)-[:CITES]->(retracted)
                    """,
                    citing_id=paper_id,
                    rid=rid.strip(),
                )

        # 3. Load Citation Contexts and HAS_CONTEXT relationships
        for occurrence, row in enumerate(dataset.contexts, start=1):
            context_id = f"ctx:{occurrence}"
            self._run(
                """
                MERGE (c:CitationContext {id: $id})
                SET c.sentence=$sentence, c.paragraph=$paragraph,
                    c.section=$section, c.timing=$timing,
                    c.matching_method=$matching_method,
                    c.matching_confidence=$matching_confidence,
                    c.source=$xml_file, c.retracted_record_id=$rid
                WITH c
                MATCH (citing:Paper {id: $citing_id})
                MATCH (retracted:Paper)
                WHERE retracted.record_id = $rid OR retracted.doi = $retracted_doi
                MERGE (citing)-[:CITES]->(retracted)
                MERGE (citing)-[:HAS_CONTEXT]->(c)
                """,
                id=context_id,
                sentence=row.get("citation_sentence", ""),
                paragraph=row.get("paragraph_text", ""),
                section=row.get("section_heading", ""),
                timing=row.get("citation_timing", ""),
                matching_method=row.get("matching_method", ""),
                matching_confidence=row.get("matching_confidence", ""),
                xml_file=row.get("xml_file", ""),
                citing_id=row.get("citing_openalex_id", ""),
                retracted_doi=row.get("retracted_doi", ""),
                rid=row.get("retracted_record_id", ""),
            )
        return self.counts()

    def counts(self) -> dict[str, int]:
        if not self.is_connected():
            return {
                "nodes": len(self.dataset.retracted) + len(self.dataset.citing) + len(self.dataset.contexts),
                "relationships": len(self.dataset.citing) + len(self.dataset.contexts) + len(self.dataset.retracted),
            }
        try:
            node_rows = self._run("MATCH (n) RETURN count(n) AS nodes")
            relationship_rows = self._run("MATCH ()-[r]->() RETURN count(r) AS relationships")
            return {
                "nodes": node_rows[0]["nodes"] if node_rows else 0,
                "relationships": relationship_rows[0]["relationships"] if relationship_rows else 0,
            }
        except Exception:
            return {"nodes": 0, "relationships": 0}

    def label_counts(self) -> list[dict[str, Any]]:
        if not self.is_connected():
            return [
                {"labels": ["Paper"], "count": len(self.dataset.retracted) + len(self.dataset.citing)},
                {"labels": ["CitationContext"], "count": len(self.dataset.contexts)},
                {"labels": ["Retraction"], "count": len(self.dataset.retracted)},
            ]
        try:
            return self._run("MATCH (n) RETURN labels(n) AS labels, count(n) AS count ORDER BY labels")
        except Exception:
            return []

    def search_papers(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        query_lower = query.lower().strip()
        if self.is_connected():
            try:
                return self._run(
                    """
                    MATCH (p:Paper)
                    WHERE toLower(coalesce(p.title,'')) CONTAINS $query_text
                       OR toLower(coalesce(p.doi,'')) CONTAINS $query_text
                    RETURN p.id AS id, p.title AS title, p.doi AS doi,
                           p.publication_date AS publication_date,
                           p.is_retracted AS is_retracted,
                           p.retraction_reasons AS retraction_reasons
                    ORDER BY p.title LIMIT $limit
                    """,
                    query_text=query_lower,
                    limit=limit,
                )
            except Exception as e:
                LOGGER.warning("Neo4j search failed, using dataset fallback: %s", e)

        # Dataset fallback
        results = []
        for p in self.dataset.retracted:
            if query_lower in (p.get("retracted_title", "") + " " + p.get("retracted_doi", "")).lower():
                results.append({
                    "id": p.get("openalex_id") or p.get("retracted_doi") or p.get("retracted_record_id"),
                    "title": p.get("retracted_title", ""),
                    "doi": p.get("retracted_doi", ""),
                    "publication_date": p.get("original_paper_date", ""),
                    "is_retracted": True,
                    "retraction_reasons": p.get("retraction_reasons", ""),
                })
        for p in self.dataset.citing:
            if query_lower in (p.get("citing_title", "") + " " + p.get("citing_doi", "")).lower():
                results.append({
                    "id": p.get("openalex_id", ""),
                    "title": p.get("citing_title", ""),
                    "doi": p.get("citing_doi", ""),
                    "publication_date": p.get("publication_date", ""),
                    "is_retracted": False,
                    "retraction_reasons": "",
                })
        return results[:limit]

    def get_subgraph_for_paper(
        self,
        record_id_or_doi: str,
        max_citing: int = 35,
        include_contexts: bool = True,
    ) -> dict[str, Any]:
        """Extract node-link subgraph for the selected retracted paper."""
        # Find retracted paper from dataset
        retracted_row = next(
            (
                r for r in self.dataset.retracted
                if record_id_or_doi in {
                    r.get("retracted_record_id", ""),
                    r.get("retracted_doi", ""),
                    r.get("openalex_id", ""),
                }
            ),
            None,
        )
        if not retracted_row:
            return {"nodes": [], "edges": [], "summary": "Paper not found."}

        retracted_id = retracted_row.get("openalex_id") or retracted_row.get("retracted_doi") or retracted_row.get("retracted_record_id")
        record_id = retracted_row.get("retracted_record_id", "")

        nodes: list[dict[str, Any]] = [
            {
                "id": retracted_id,
                "label": "Retracted Paper",
                "title": retracted_row.get("retracted_title", "Retracted Paper"),
                "doi": retracted_row.get("retracted_doi", ""),
                "date": retracted_row.get("original_paper_date", ""),
                "retraction_date": retracted_row.get("retraction_date", ""),
                "reasons": retracted_row.get("retraction_reasons", ""),
                "journal": retracted_row.get("journal", ""),
                "node_type": "retracted",
                "size": 28,
                "color": "#e53e3e",
            },
            {
                "id": f"{retracted_id}:retraction",
                "label": "Retraction Notice",
                "title": f"Retracted on {retracted_row.get('retraction_date', 'Unknown')}",
                "date": retracted_row.get("retraction_date", ""),
                "reasons": retracted_row.get("retraction_reasons", ""),
                "node_type": "retraction",
                "size": 16,
                "color": "#c53030",
            },
        ]

        edges: list[dict[str, Any]] = [
            {
                "source": retracted_id,
                "target": f"{retracted_id}:retraction",
                "type": "RETRACTED_FOR",
                "label": "RETRACTED_FOR",
            }
        ]

        # Find citing papers connected to this retracted paper
        all_citing = [
            c for c in self.dataset.citing
            if record_id in c.get("retracted_record_ids", "").split(";")
        ]
        all_contexts = [
            ctx for ctx in self.dataset.contexts
            if ctx.get("retracted_record_id") == record_id
        ]
        context_by_citing: dict[str, list[dict[str, str]]] = {}
        for ctx in all_contexts:
            context_by_citing.setdefault(ctx.get("citing_openalex_id", ""), []).append(ctx)

        # Prioritize citing papers that have full-text citation contexts
        with_ctx = [c for c in all_citing if c.get("openalex_id") in context_by_citing]
        without_ctx = [c for c in all_citing if c.get("openalex_id") not in context_by_citing]
        selected_citing = (with_ctx + without_ctx)[:max_citing]

        for citing in selected_citing:
            c_id = citing.get("openalex_id")
            has_contexts = c_id in context_by_citing
            nodes.append({
                "id": c_id,
                "label": "Citing Paper",
                "title": citing.get("citing_title", "Untitled Citing Paper"),
                "doi": citing.get("citing_doi", ""),
                "date": citing.get("publication_date", ""),
                "authors": citing.get("citing_authors", ""),
                "journal": citing.get("citing_journal", ""),
                "xml_status": citing.get("xml_status", ""),
                "node_type": "citing",
                "has_context": has_contexts,
                "size": 18 if has_contexts else 14,
                "color": "#1976a3" if has_contexts else "#627d98",
            })
            # Citing paper cites retracted paper
            edges.append({
                "source": c_id,
                "target": retracted_id,
                "type": "CITES",
                "label": "CITES",
            })

            # Add Context nodes if enabled
            if include_contexts and has_contexts:
                for idx, ctx in enumerate(context_by_citing[c_id][:3], start=1):
                    ctx_id = f"{c_id}:ctx:{idx}"
                    timing = ctx.get("citation_timing", "DATE_UNCERTAIN")
                    timing_color = "#e67e22" if timing == "AFTER_RETRACTION" else "#27ae60"
                    nodes.append({
                        "id": ctx_id,
                        "label": "Citation Context",
                        "title": f"Context ({timing})",
                        "sentence": ctx.get("citation_sentence", ""),
                        "paragraph": ctx.get("paragraph_text", ""),
                        "section": ctx.get("section_heading", ""),
                        "timing": timing,
                        "source": ctx.get("xml_file", ""),
                        "node_type": "context",
                        "size": 12,
                        "color": timing_color,
                    })
                    edges.append({
                        "source": c_id,
                        "target": ctx_id,
                        "type": "HAS_CONTEXT",
                        "label": "HAS_CONTEXT",
                        "timing": timing,
                    })

        summary = (
            f"Retracted publication '{retracted_row.get('retracted_title')[:60]}...' "
            f"connected to {len(all_citing)} discovered citing publications and "
            f"{len(all_contexts)} extracted citation contexts."
        )
        return {
            "retracted_paper": retracted_row,
            "nodes": nodes,
            "edges": edges,
            "summary": summary,
            "counts": {
                "citing_papers": len(all_citing),
                "contexts": len(all_contexts),
                "before_retraction": sum(1 for c in all_contexts if c.get("citation_timing") == "BEFORE_RETRACTION"),
                "after_retraction": sum(1 for c in all_contexts if c.get("citation_timing") == "AFTER_RETRACTION"),
            },
        }

    def paths(self, paper_id: str, hops: int = 2) -> list[dict[str, Any]]:
        hops = max(1, min(hops, 4))
        if self.is_connected():
            try:
                rows = self._run(
                    f"""
                    MATCH path=(start:Paper {{id: $paper_id}})-[*1..{hops}]-(target)
                    WHERE target:Paper OR target:CitationContext
                    RETURN [node IN nodes(path) | {{id: node.id, labels: labels(node), title: coalesce(node.title, node.sentence, node.id)}}] AS nodes,
                           [rel IN relationships(path) | type(rel)] AS relationships
                    LIMIT 50
                    """,
                    paper_id=paper_id,
                )
                if rows:
                    return rows
            except Exception as e:
                LOGGER.warning("Neo4j path query failed, using dataset fallback: %s", e)

        # Fallback path traversal from dataset
        subgraph = self.get_subgraph_for_paper(paper_id, max_citing=20, include_contexts=True)
        paths = []
        retracted_node = next((n for n in subgraph["nodes"] if n["node_type"] == "retracted"), None)
        if not retracted_node:
            return []

        for edge in subgraph["edges"]:
            if edge["type"] == "CITES":
                citing_node = next((n for n in subgraph["nodes"] if n["id"] == edge["source"]), None)
                if citing_node:
                    paths.append({
                        "nodes": [
                            {"id": retracted_node["id"], "labels": ["Paper"], "title": retracted_node["title"]},
                            {"id": citing_node["id"], "labels": ["Paper"], "title": citing_node["title"]},
                        ],
                        "relationships": ["CITES"],
                    })
                    # Add context 2-hop paths
                    for c_edge in subgraph["edges"]:
                        if c_edge["source"] == citing_node["id"] and c_edge["type"] == "HAS_CONTEXT":
                            ctx_node = next((n for n in subgraph["nodes"] if n["id"] == c_edge["target"]), None)
                            if ctx_node:
                                paths.append({
                                    "nodes": [
                                        {"id": retracted_node["id"], "labels": ["Paper"], "title": retracted_node["title"]},
                                        {"id": citing_node["id"], "labels": ["Paper"], "title": citing_node["title"]},
                                        {"id": ctx_node["id"], "labels": ["CitationContext"], "title": ctx_node.get("sentence", "")[:60]},
                                    ],
                                    "relationships": ["CITES", "HAS_CONTEXT"],
                                })
        return paths[:50]
