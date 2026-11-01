# RetractTrace: Comprehensive System Implementation Details

This document provides a complete technical reference for the architecture, mathematical formulations, algorithms, data structures, and execution workflows implemented in **RetractTrace**.

---

## 1. System Architecture & High-Level Design

RetractTrace is a non-agentic, deterministic research intelligence system. It combines full-text JATS XML parsing, knowledge graph traversal (Neo4j), dense vector indexing (FAISS), and multi-objective ranking to detect, classify, and explain post-retraction citation patterns.

```
                              ┌───────────────────────────────────┐
                              │     Crossref Retraction Watch     │
                              │          (67.8 MB CSV)            │
                              └─────────────────┬─────────────────┘
                                                │ 100 Selected Papers
                                                ▼
                              ┌───────────────────────────────────┐
                              │       OpenAlex Graph API          │
                              │   (915 Discovered Citing Works)   │
                              └─────────────────┬─────────────────┘
                                                │ Resolve PMCIDs & OA License
                                                ▼
                              ┌───────────────────────────────────┐
                              │      NCBI Europe PMC Service      │
                              │     (104 JATS XML Full-Texts)     │
                              └─────────────────┬─────────────────┘
                                                │ Reference Context Extraction
                                                ▼
                              ┌───────────────────────────────────┐
                              │       148 Citation Contexts       │
                              │ (94 Pre-Retraction, 54 Post-Retr) │
                              └─────────────────┬─────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│  Knowledge Graph │                  │ Intent Classifier│                  │ Retrieval Engine │
│     (Neo4j)      │                  │  (Rule + Context)│                  │  (4 Retrievers)  │
├──────────────────┤                  ├──────────────────┤                  ├──────────────────┤
│ 1,263 Nodes      │                  │ 6 Categories     │                  │ BM25 (Lexical)   │
│ 1,163 Edges      │                  │ 2.5x Sentence Wt │                  │ Dense (FAISS)    │
│ Subgraph Tracing │                  │ 1.0x Para Wt     │                  │ Hybrid (RRF)     │
│ Dual-Mode Fallbk │                  │ Human Annotation │                  │ Graph-Enhanced   │
└────────┬─────────┘                  └────────┬─────────┘                  └────────┬─────────┘
         │                                      │                                      │
         └──────────────────────────────────────┼──────────────────────────────────────┘
                                                ▼
                              ┌───────────────────────────────────┐
                              │      FastAPI REST Application     │
                              │         (Port 8000)               │
                              └─────────────────┬─────────────────┘
                                                ▼
                              ┌───────────────────────────────────┐
                              │        Streamlit Frontend         │
                              │       (6 Workspaces / 8501)       │
                              └───────────────────────────────────┘
```

---

## 2. Data Pipeline & Processing Methodology

### 2.1 Retraction Watch Ingestion (`src/data_acquisition/retraction_watch.py`)
- Reads the canonical Crossref Retraction Watch CSV.
- Filters for records containing valid DOIs, titles, retraction dates, and original publication dates.
- Selects the top 100 benchmark retracted papers across high-impact biomedical and materials science disciplines.
- Persisted at `data/processed/selected_retracted_papers.csv`.

### 2.2 Citing Work Discovery (`src/data_acquisition/openalex.py`)
- Queries OpenAlex for all publications citing each retracted paper DOI.
- Harvests bibliographic metadata: title, publication year, publication date, journal, citation count, and author list.
- Discovers **915 citing publications**, persisted at `data/processed/citing_papers.csv`.

### 2.3 PMC XML Resolution & Parsing (`src/preprocessing/xml_parser.py`)
- Maps citing paper identifiers to PubMed Central IDs (PMCIDs) via NCBI ID Converter API.
- Evaluates open-access text-mining licenses to ensure copyright compliance.
- Downloads **104 full-text JATS XML documents** into `data/raw/pmc_xml/`.
- Extracts XML references (`<ref-list>`), identifies the target retracted paper by matching DOI, title, or authors, and locates citing `<xref>` tags within the body text.
- Extracts the exact **citation sentence** and the **surrounding paragraph**.
- Yields **148 extracted citation contexts** (94 pre-retraction, 54 post-retraction), persisted at `data/processed/citation_contexts.csv`.

---

## 3. Knowledge Graph Engine & Neo4j Modeling

### 3.1 Graph Schema
The graph models scholarly entities and their provenance:
1. **`(:Paper)`**:
   - `paper_id` (String, Unique Primary Key, e.g. OpenAlex ID or DOI)
   - `doi` (String)
   - `title` (String)
   - `publication_date` (String, YYYY-MM-DD)
   - `is_retracted` (Boolean)
   - `citation_count` (Integer)
2. **`(:Retraction)`**:
   - `retraction_id` (String, Unique Primary Key)
   - `retraction_date` (String, YYYY-MM-DD)
   - `reason` (String, e.g., "Concerns/Issues About Data; Unreliable Results")
3. **`(:CitationContext)`**:
   - `context_id` (String, Unique Primary Key)
   - `sentence` (String)
   - `paragraph` (String)
   - `section` (String)
   - `is_post_retraction` (Boolean)
   - `intent` (String)

### 3.2 Relationships
- `(:Paper)-[:CITES {citing_paper_id, cited_paper_id}]->(:Paper)`
- `(:Paper)-[:HAS_CONTEXT]->(:CitationContext)`
- `(:Paper)-[:RETRACTED_FOR]->(:Retraction)`

### 3.3 Path Reasoning & Dual-Mode Fallback (`src/knowledge_graph/service.py`)
- **Direct Path Traversal:**
  $$\text{Retracted Paper } (P_r) \xrightarrow{\text{CITES}} \text{Citing Paper } (P_c) \xrightarrow{\text{HAS\_CONTEXT}} \text{Citation Context } (C)$$
- **Dual-Mode Engine:**
  - *Primary:* Cypher queries over Bolt (`bolt://localhost:7687`) or Neo4j Aura.
  - *Fallback:* Zero-overhead in-memory NetworkX/dataset graph engine if Neo4j is offline, guaranteeing continuous availability without runtime crashes.

---

## 4. Multi-Objective Retrieval Architecture (`src/retrieval/engine.py`)

RetractTrace implements four distinct retrieval strategies:

### 4.1 Lexical BM25 Retrieval
- Tokenizes query and corpus documents (citation sentences, paragraphs, and titles).
- Uses Rank-BM25 with parameters $k_1 = 1.5, b = 0.75$.

### 4.2 Dense Semantic Retrieval (FAISS)
- Employs `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- Builds a FAISS `IndexFlatIP` (inner product on normalized vectors $\rightarrow$ cosine similarity).

### 4.3 Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
- Combines sparse BM25 and dense semantic rankings without arbitrary score normalization:
  $$RRF(d) = \frac{1}{60 + r_{bm25}(d)} + \frac{1}{60 + r_{dense}(d)}$$

### 4.4 Graph-Enhanced Retrieval
Combines textual semantics with explicit graph topology and citation temporal risk:
$$S_{graph}(d) = w_{dense} S_{dense}(d) + w_{bm25} S_{bm25}(d) + w_{cite} \mathbb{I}_{cite}(d) + w_{post} \mathbb{I}_{post\_retract}(d) + w_{rel} \mathbb{I}_{reliance}(d)$$

**Standard Configured Weights:**
- $w_{dense} = 0.40$: Dense semantic relevance
- $w_{bm25} = 0.20$: Sparse lexical matching
- $w_{cite} = 0.15$: Direct citation relationship in the knowledge graph
- $w_{post} = 0.15$: Post-retraction citation risk (paper published after the retraction date)
- $w_{rel} = 0.10$: Citation classified as `SUPPORTING_OR_RELIANCE` or `METHODOLOGICAL`

Each result emits explicit ranking factor evidence:
```json
{
  "dense_component": 0.324,
  "bm25_component": 0.182,
  "citation_link_bonus": 0.150,
  "post_retraction_risk_bonus": 0.150,
  "intent_reliance_bonus": 0.100,
  "total_graph_score": 0.906
}
```

---

## 5. Citation Intent Classification Engine (`src/classification/engine.py`)

### 5.1 Scholarly Intent Categories
1. `SUPPORTING_OR_RELIANCE`: Citing paper directly builds upon, adopts, or relies upon the findings, data, or conclusions of the retracted paper.
2. `BACKGROUND`: General literature survey, historical overview, or passing introduction mention.
3. `METHODOLOGICAL`: Adoption, adaptation, or reference to experimental protocols, reagents, or software tools.
4. `CRITICAL`: Dispute, critique, failure to replicate, or contradiction of the cited findings.
5. `RETRACTION_ACKNOWLEDGEMENT`: Explicit mention that the cited paper was retracted, withdrawn, or has an erratum/expression of concern.
6. `UNCLEAR`: Neutral citation lacking explicit linguistic markers to assign an affirmative category.

### 5.2 Contextual Weighting Algorithm
The classifier analyzes both the citation sentence and the surrounding paragraph with asymmetric weighting:
- **Citation Sentence Weight:** $2.5\times$
- **Surrounding Paragraph Weight:** $1.0\times$
- **Negation Handling:** Evaluates preceding tokens within a 4-word window for explicit negations ("did not support", "failed to confirm", "unsupported") to prevent false positives.
- **Calibrated Confidences:** Confidence is bounded between $[0.40, 0.95]$ based on evidence match density and linguistic clarity.

### 5.3 Human Annotation System
- Dedicated ground-truth store at `data/annotations/citation_context_annotations.csv`.
- Fields: `context_id`, `citing_paper_id`, `retracted_paper_id`, `citation_sentence`, `surrounding_paragraph`, `human_label`, `annotator_id`, `annotation_notes`, `timestamp`.
- Updates preserve existing human labels without overwriting or generating synthetic ground truth.

---

## 6. Explainability & 14-Section Markdown Report

Every investigated paper can be exported as a comprehensive 14-section investigation report (`src/explainability/service.py`):
1. **Executive Summary**: Overview of retracted work and citation count.
2. **Retracted Publication Profile**: Full title, DOI, authors, journal, and publication date.
3. **Retraction Metadata & Reason**: Retraction notice date, elapsed time, and retraction reasons.
4. **Citing Literature Overview**: Total citing papers, discovered PMC open-access articles.
5. **Temporal Risk Classification**: Pre-retraction vs. post-retraction breakdown.
6. **Citation Context Extraction Provenance**: JATS XML file sources and extraction confidence.
7. **Intent Classification Breakdown**: 6-category intent distribution.
8. **Knowledge Graph Neighborhood Topology**: Node degrees, connected subgraphs, and relationship counts.
9. **Citation Path Reasoning**: Explicit directional chains (`CITES` $\rightarrow$ `HAS_CONTEXT`).
10. **Post-Retraction Reliance Risk Assessment**: Highlighted papers that rely on invalidated science.
11. **Retrieval Relevance & Ranking Factor Audit**: Why papers were retrieved by BM25, Dense, and Graph engines.
12. **Limitations & Missing Evidence**: Unparsed paywalled articles and open-access licensing constraints.
13. **Responsible Use & Non-Causality Disclaimers**: Clarifying that citation does not imply fraud.
14. **Audit Metadata & Provenance**: Generation timestamp, system version, and reproducible environment info.

---

## 7. Empirical Evaluation & Benchmarks (`src/evaluation/metrics.py`)

### 7.1 Benchmark Test Queries
Evaluated across 5 documented domain queries:
1. `"cancer metabolism ovarian"`
2. `"nanoparticles drug delivery"`
3. `"microRNA expression tumor pancreatic"`
4. `"biomarkers diagnostic prognosis"`
5. `"electrocatalyst materials energy"`

### 7.2 Performance Comparison ($K=5$)
| Metric | BM25 | Dense (FAISS) | Hybrid (RRF) | Graph-Enhanced |
|---|---|---|---|---|
| **NDCG@5** | 0.4216 | 0.2852 | 0.3986 | **0.5482** |
| **Precision@5** | 0.6000 | 0.4000 | 0.6000 | **0.8000** |
| **Recall@5** | 0.6667 | 0.4444 | 0.6667 | **0.8889** |
| **MRR** | 0.8333 | 0.6000 | 0.8333 | **1.0000** |
| **Latency** | **4.2 ms** | 28.5 ms | 32.1 ms | 31.3 ms |

---

## 8. Verified Acceptance Case Studies

### Acceptance Case 1: Ovarian Cancer High-Volume Case
- **Record ID:** 70462 | **DOI:** `10.1021/acscentsci.8b00050`
- **Topic:** High-throughput metabolic profiling in ovarian cancer.
- **Data:** 50 citing publications, 6 PMC XMLs, 7 citation contexts.
- **Key Finding:** 4 post-retraction citations; 2 classified as `SUPPORTING_OR_RELIANCE` adopting the assay without retraction awareness.

### Acceptance Case 2: Pancreatic Cancer Post-Retraction Reliance
- **Record ID:** 66404 | **DOI:** `10.1093/abbs/gmz096`
- **Topic:** MicroRNA dysregulation in pancreatic ductal adenocarcinoma.
- **Data:** 18 citing publications, 2 PMC XMLs, 2 citation contexts.
- **Key Finding:** Both contexts are post-retraction; citing authors cite target miRNA expression profiles as verified biological evidence.

### Acceptance Case 3: Materials Science Cross-Disciplinary Case
- **Record ID:** 53194 | **DOI:** `10.1021/acsami.9b23052`
- **Topic:** Electrocatalytic nanomaterials for energy conversion.
- **Data:** 24 citing publications, 3 PMC XMLs, 4 citation contexts.
- **Key Finding:** 1 `METHODOLOGICAL` context adopting synthesis parameters post-retraction; 1 `BACKGROUND` reference in an energy review.
