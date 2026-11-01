# RetractTrace: Explainable Knowledge-Graph IR for Retracted Scientific Literature

RetractTrace is an explainable, knowledge-graph-based information retrieval and research intelligence platform for investigating the continued influence of retracted scientific publications. 

RetractTrace provides a **non-agentic, deterministic research intelligence workflow** that links bibliographic metadata, full-text citation contexts from open-access PMC articles, graph-based citation networks, and multi-objective ranking to trace, classify, and explain post-retraction citation patterns.

---

## 1. Research Objectives & Alignment

Scientific retractions exist to correct the literature, yet thousands of studies continue to be cited uncritically years after retraction. RetractTrace addresses this critical problem by answering:
1. **Which papers continue to cite retracted research?**
2. **What is the specific citation context and scientific intent?** (e.g., direct reliance vs. background reference vs. retraction awareness)
3. **Were the citations made before or after the retraction date?**
4. **How do citation chains propagate through scholarly knowledge graphs?**
5. **How can retrieval systems explain ranking decisions using graph and textual evidence?**

### Sustainable Development Goals (SDG) Alignment:
- **SDG 9, Target 9.5 (Enhance Scientific Research & Innovation):** Protects the integrity and reproducibility of scientific research by systematically exposing reliance on invalid findings.
- **SDG 16, Target 16.6 (Develop Effective, Accountable, and Transparent Institutions):** Ensures transparency and accountability in scholarly publishing with verifiable provenance chains and explicit distinction between correlation and causation.

---

## 2. System Architecture & Dataset Provenance

```
RetractTrace/
├── data/
│   ├── raw/
│   │   ├── retraction_watch/      # Source Retraction Watch CSV (preserved)
│   │   └── pmc_xml/               # 104 genuine open-access PMC JATS XML files
│   ├── interim/                   # Cache: API responses, unavailable articles, failed matches
│   ├── processed/                 # Canonical Phase 1 datasets
│   │   ├── selected_retracted_papers.csv  (100 retracted papers)
│   │   ├── citing_papers.csv              (915 citing publications)
│   │   ├── citation_contexts.csv          (148 extracted citation contexts)
│   │   └── classified_contexts.csv        (148 classified contexts with confidence)
│   └── annotations/
│       └── citation_context_annotations.csv (Persistent human validation ground truth)
├── src/
│   ├── data_acquisition/          # Resumable Retraction Watch & NCBI PMC acquisition
│   ├── preprocessing/             # JATS XML parsing & reference context extraction
│   ├── classification/            # 6-category citation intent engine & annotation store
│   ├── knowledge_graph/           # Neo4j Cypher loading, subgraph querying, & graph fallback
│   ├── retrieval/                 # BM25, Dense (FAISS), Hybrid (RRF), & Graph-Enhanced
│   ├── explainability/            # Provenance tracing & 14-section Markdown report generation
│   ├── evaluation/                # NDCG, MAP, MRR, latency benchmarks, & audit metrics
│   ├── api/                       # FastAPI REST service
│   └── utils/                     # Config management & logging
├── frontend/                      # Streamlit 6-workspace research intelligence app
│   └── app.py
├── tests/                         # Automated test suites (test_core.py, test_platform.py)
└── scripts/                       # Repeatable execution scripts
```

### Dataset Summary
- **100 Retracted Papers:** Spanning oncology, biochemistry, genetics, materials science, and COVID-19.
- **915 Citing Publications:** Discovered via OpenAlex and Crossref.
- **104 Full-Text PMC JATS XMLs:** Resolved and downloaded under open-access reuse licenses.
- **148 Citation Contexts:** Extracted directly from XML reference tags and body paragraphs.
  - **94 Before-Retraction Citations (63.5%)**
  - **54 After-Retraction Citations (36.5%)**

---

## 3. Core Capabilities

### A. Interactive Plotly Knowledge Graph Explorer
- **Interactive Node-Link Visualization:** Rendered using Plotly Graph Objects with customizable layout spring algorithms.
- **Multi-Type Entity Modeling:** Distinct visual styles for:
  - 🔴 **Retracted Paper** (Central red node, large marker)
  - 🔵 **Citing Paper** (Blue node, pre- or post-retraction marker styling)
  - 🟢 **Citation Context** (Green context node with explicit intent and timing)
  - 🟡 **Retraction Reason** (Amber category node)
- **Path Reasoning Engine:** Explores and explains explicit directional graph paths:
  $$\text{Retracted Paper} \xrightarrow{\text{CITES}} \text{Citing Paper} \xrightarrow{\text{HAS\_CONTEXT}} \text{Citation Context}$$
- **Dual-Mode Graph Service:** Connects directly to Neo4j via Bolt (`bolt://localhost:7687` or Neo4j Aura). If Neo4j is offline or disconnected, an automatic, zero-overhead in-memory dataset fallback maintains full functionality.

### B. Multi-Objective Graph-Enhanced Retrieval
RetractTrace implements four independent retrieval pipelines:
1. **Lexical BM25:** Tokenized BM25 ranking across citation sentences, paragraphs, and titles.
2. **Dense Semantic Retrieval:** FAISS FlatL2 vector index using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings).
3. **Hybrid Retrieval (RRF):** Reciprocal Rank Fusion combining BM25 and Dense ranks ($k=60$):
   $$RRF(d) = \frac{1}{60 + r_{bm25}(d)} + \frac{1}{60 + r_{dense}(d)}$$
4. **Graph-Enhanced Multi-Objective Retrieval:** Incorporates lexical relevance, dense semantics, direct citation connections, temporal risk, and reliance intent:
   $$S_{graph}(d) = 0.40 \cdot S_{dense} + 0.20 \cdot S_{bm25} + 0.15 \cdot \mathbb{I}_{cite} + 0.15 \cdot \mathbb{I}_{post\_retract} + 0.10 \cdot \mathbb{I}_{reliance}$$

### C. Citation Intent Classification & Human Annotation
- **Six Scholarly Categories:**
  1. `SUPPORTING_OR_RELIANCE`: Direct adoption of findings, protocols, or affirmative reliance.
  2. `BACKGROUND`: General literature survey or introduction mention.
  3. `METHODOLOGICAL`: Usage or modification of experimental protocols or tools.
  4. `CRITICAL`: Disagreement, contradiction, or questioning of results.
  5. `RETRACTION_ACKNOWLEDGEMENT`: Explicit awareness that the cited paper is retracted.
  6. `UNCLEAR`: Neutral or ambiguous citation lacking sufficient context.
- **Contextual Weighting:** Citation sentence receives **2.5× weight** over surrounding paragraph context (1.0×), with explicit phrase negation handling.
- **Persistent Human Annotation Workflow:** Annotators can inspect citation sentences and paragraphs side-by-side and record ground truth directly to `data/annotations/citation_context_annotations.csv`.

### D. Evidence-Backed Explainability & Reporting
- Every search result displays:
  - Exact citation sentence and surrounding paragraph provenance.
  - Source PMC XML file and PMCID.
  - Chronological timeline (publication date vs. retraction date).
  - Component ranking factor breakdown.
- **14-Section Markdown Report Export:** Generates publication-ready investigation briefs with executive summaries, graph topology, temporal risk classification, and ethical considerations.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10 to 3.13
- Git
- (Optional) Docker or local Neo4j 5.x instance

### 1. Clone & Set Up Virtual Environment
```powershell
git clone <repository_url>
cd KGIRS_Mini_Project

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create or edit `.env` in the repository root:
```ini
# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
NEO4J_DATABASE=neo4j

# NCBI PMC Identification (Required for PMC API rate limits)
PMC_EMAIL=your_email@domain.com
PMC_TOOL=RetractTrace
PMC_API_KEY=
```
*(Note: If Neo4j is not running, the application will automatically fall back to the in-memory dataset graph engine.)*

### 3. Load the Knowledge Graph (Optional if using Neo4j)
To populate Neo4j with the 1,263 nodes and 1,163 relationships:
```powershell
python -m scripts.build_knowledge_graph
```

---

## 5. Running the Application

### Start the FastAPI Backend
```powershell
python -m uvicorn src.api.app:app --port 8000 --reload
```
Interactive Swagger API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

Key Endpoints:
- `GET /health` & `GET /stats`: System health and dataset metrics.
- `GET /search`: Multi-engine search (`method=bm25|dense|hybrid|graph_enhanced`).
- `GET /retrieval/compare`: Side-by-side comparison across all 4 retrieval methods.
- `GET /graph/subgraph/{paper_id}`: Subgraph nodes and edges for Plotly visualization.
- `GET /graph/paths/{paper_id}`: Explicit directional paths between retracted and citing papers.
- `GET /evaluation/retrieval`: NDCG, MAP, MRR benchmark across standard test queries.
- `POST /annotations/save`: Persist human ground-truth labels.

### Start the Streamlit Frontend
In a separate terminal:
```powershell
python -m streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) to access the 6 workspaces:
1. **Overview:** High-level platform statistics, temporal breakdown, and system architecture.
2. **Paper Discovery:** Search and inspect the 100 retracted publications and their retraction notices.
3. **Citation Intelligence:** Inspect 148 citation contexts, review rule predictions, perform human annotations, and export 14-section investigation reports.
4. **Knowledge Graph Explorer:** Interactive Plotly node-link graph visualization, dynamic neighborhood expansion, and path reasoning.
5. **Retrieval & Explainability:** Side-by-side execution of BM25, Dense, Hybrid, and Graph-Enhanced retrieval with factor breakdown.
6. **Evaluation & Benchmarks:** Comprehensive benchmark dashboard displaying NDCG@K, MRR, latency comparisons, and provenance audits.

---

## 6. Automated Testing

RetractTrace includes a comprehensive automated test suite covering unit operations, API endpoints, retrieval algorithms, graph traversal, and full acceptance case verification:

```powershell
# Run core unit tests
python -m unittest tests/test_core.py

# Run end-to-end platform validation (all 11 test cases)
python -m unittest tests/test_platform.py
```

---

## 7. Acceptance Case Studies

The platform has been validated against three real-world acceptance cases representing distinct scientific domains:

### Case 1: High-Volume Biomedical Case (Ovarian Cancer)
- **Record ID:** 70462
- **DOI:** `10.1021/acscentsci.8b00050`
- **Topic:** High-throughput metabolic screening in ovarian cancer.
- **Citing Articles:** 50 citing publications discovered.
- **Full-Text Contexts:** 6 open-access PMC XMLs parsed; 7 citation contexts extracted.
- **Key Finding:** 4 citations occurred **after** retraction; 2 classified as `SUPPORTING_OR_RELIANCE` affirming the primary screening protocol without citing the retraction notice.

### Case 2: Post-Retraction Reliance Case (Pancreatic Cancer)
- **Record ID:** 66404
- **DOI:** `10.1093/abbs/gmz096`
- **Topic:** MicroRNA dysregulation in pancreatic ductal adenocarcinoma.
- **Citing Articles:** 18 citing publications discovered.
- **Full-Text Contexts:** 2 open-access PMC XMLs parsed; 2 citation contexts extracted.
- **Key Finding:** Demonstrates persistent post-retraction citation where citing authors adopted the target miRNA expression patterns as established biological fact without acknowledging the retraction.

### Case 3: Materials Science Cross-Disciplinary Case
- **Record ID:** 53194
- **DOI:** `10.1021/acsami.9b23052`
- **Topic:** Functionalized nanomaterials for electrocatalysis.
- **Citing Articles:** 24 citing publications discovered.
- **Full-Text Contexts:** 3 open-access PMC XMLs parsed; 4 citation contexts extracted.
- **Key Finding:** Mixed intent citations including 1 `METHODOLOGICAL` context adopting nanomaterial synthesis parameters after retraction and 1 `BACKGROUND` reference in an energy storage review.

---

## 8. Evaluation Benchmark Summary

Retrieval performance benchmarked on 5 documented scholarly queries across the genuine dataset:

| Retrieval Method | Precision@5 | Recall@5 | MRR | NDCG@5 | Latency (ms) |
|---|---|---|---|---|---|
| **BM25** | 0.6000 | 0.6667 | 0.8333 | 0.4216 | **4.2 ms** |
| **Dense (FAISS)** | 0.4000 | 0.4444 | 0.6000 | 0.2852 | 28.5 ms |
| **Hybrid (RRF)** | 0.6000 | 0.6667 | 0.8333 | 0.3986 | 32.1 ms |
| **Graph-Enhanced** | **0.8000** | **0.8889** | **1.0000** | **0.5482** | 31.3 ms |

*Explanation:* Graph-Enhanced Retrieval achieves the highest NDCG@5 and MRR because it promotes publications that cite retracted papers post-retraction with reliance intent, addressing the core information need of research integrity investigations.

---

## 9. Scientific Limitations & Responsible Use

1. **Non-Causal Association:** A graph edge (`CITES`) or textual mention indicates citation linkage, **not** proof of scientific misconduct, fraud, or intentional reliance.
2. **Open-Access Full-Text Coverage:** Full-text context extraction is constrained to articles available under open-access reuse licenses in Europe PMC / PMC Open Access Subset (104 of 915 citing publications).
3. **Calibrated Probabilities:** Rule-based classification confidence reflects heuristic keyword match strength, not calibrated Bayesian posterior probabilities.
4. **Human Label Distinction:** Rule-based predictions are kept strictly segregated from human annotations in `data/annotations/citation_context_annotations.csv`.
