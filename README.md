# RetractTrace

RetractTrace is an explainable, knowledge-graph-based information retrieval
system for identifying research papers that cite or may rely on retracted
scientific findings.

Phase 1 provides a resumable data acquisition and XML processing pipeline.
The current platform layer adds annotation-ready classification, local
retrieval, explainability, optional Neo4j Aura loading, FastAPI endpoints, and
a Streamlit interface over the genuine Phase 1 outputs.

## Objectives

- Integrate retraction metadata and scientific full-text sources.
- Extract papers, references, citations, and citation contexts.
- Classify the relationship between citations and retracted findings.
- Build a Neo4j knowledge graph with traceable evidence.
- Support BM25, dense, hybrid, and graph-aware retrieval.
- Provide explainable results and evaluation workflows.
- Expose the system through a FastAPI backend and Streamlit frontend.

## Architecture

The project separates source data, processing stages, reusable modules, and
user-facing applications:

```text
data/          Dataset storage separated by lifecycle stage
src/           Reusable application and domain modules
frontend/      Streamlit research intelligence interface
models/        Model artifacts and checkpoints
notebooks/     Exploratory analysis and experiments
tests/         Automated tests
configs/       Application configuration files
scripts/       Repeatable project utility scripts
```

### Source modules

- `data_acquisition/`: Retraction Watch and PMC acquisition workflows.
- `preprocessing/`: XML parsing, normalization, and document preparation.
- `classification/`: Citation-context classification components.
- `knowledge_graph/`: Neo4j schemas, loading, and graph access.
- `retrieval/`: Sparse, dense, hybrid, and graph-based retrieval.
- `explainability/`: Evidence tracing and result explanations.
- `evaluation/`: Retrieval and classification evaluation.
- `api/`: FastAPI application and API contracts.
- `utils/`: Shared configuration and utility functions.

Each source directory is an independent Python package boundary.

## Phase 1 data pipeline

The Phase 1 pipeline is implemented in
[`scripts/run_data_pipeline.py`](./scripts/run_data_pipeline.py). It loads up
to 100 usable original-paper identifiers from the Retraction Watch CSV,
discovers citing works with OpenAlex, resolves PMCIDs with NCBI services,
checks PMC reuse licences, downloads eligible JATS XML, and extracts
reference-level citation contexts.

Run it from the repository root:

```powershell
python -m scripts.run_data_pipeline
```

The input CSV is preserved. API responses are cached in `data/interim/`, XML
is stored in `data/raw/pmc_xml/`, and the required outputs are written to
`data/processed/`. Existing cached responses and XML files are reused on
restart. Set `PMC_EMAIL`, `PMC_API_KEY`, and optionally `OPENALEX_EMAIL` in
`.env` to identify requests and use the applicable service limits.

The pipeline also writes `data/interim/unavailable_articles.csv` and
`data/interim/failed_matches.csv` so unresolved identifiers, unavailable
full text, licence exclusions, and unmatched references remain explicit.

## Platform layer

Create an annotation-ready file without adding labels:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_annotation_dataset
```

Run the rule-based classification baseline:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_classification
```

Load the actual configured Neo4j Aura instance:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_knowledge_graph
```

Run the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --reload
```

Run the Streamlit interface:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

The local BM25 search and rule-based citation classifier operate directly on
the stored Phase 1 records. Dense retrieval requires the declared
Sentence-Transformers and FAISS dependencies and a locally available model.
Neo4j operations use the existing Aura variables in `.env`; the application
does not substitute a local database and reports missing connectivity
explicitly.

The Streamlit interface is organized into five workflow sections: Overview,
Paper Discovery, Citation Intelligence, Knowledge Graph Explorer, and
Retrieval & Explainability. Paper investigation reports are exported from the
Citation Intelligence section as Markdown using the selected paper's actual
metadata and extracted contexts. Classification predictions remain distinct
from human validation, and no evaluation score is shown without genuine
labels.

Classification predictions are not human-validated labels. No accuracy or
retrieval score is reported until genuine annotations or relevance judgments
exist. Graph paths describe potential citation propagation, not causal
influence or scientific reliance.

## Research alignment

RetractTrace supports **SDG 9, Target 9.5** by making research-reliability
evidence searchable and traceable, and **SDG 16, Target 16.6** by preserving
provenance and uncertainty in scholarly relationships. The implementation
does not claim geospatial, multimedia, or distributed retrieval capabilities.

## Setup

### Windows virtual environment

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents script activation, use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set credentials locally when integrations
are introduced. Never commit `.env` or real credentials.

### Run the initial entry point

```powershell
python main.py
```

Use the Phase 1 pipeline or platform commands above for execution.

### Git initialization

For a new local checkout:

```powershell
git init
git add .
git commit -m "Initialize RetractTrace architecture"
```

## Current scope

The system does not fabricate labels, retrieval judgments, graph edges based
only on citation, unavailable full text, or successful external integrations.
