"""RetractTrace research intelligence workflow."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.api.app import dense_resource, resources
from src.classification.engine import CATEGORIES, CitationClassifier
from src.evaluation.metrics import classification_metrics
from src.knowledge_graph.service import GraphService
from src.preprocessing.dataset import Dataset, dataset_statistics, read_csv
from src.retrieval.engine import reciprocal_rank_fusion
from src.utils.config import settings


st.set_page_config(page_title="RetractTrace", page_icon="R", layout="wide")

st.markdown(
    """
    <style>
    :root { --navy:#102a43; --slate:#486581; --blue:#1976a3; --line:#d9e2ec; }
    .stApp { background:#f7f9fb; color:var(--navy); }
    [data-testid="stSidebar"] { background:#102a43; }
    [data-testid="stSidebar"] * { color:#f0f4f8 !important; }
    .hero { background:linear-gradient(115deg,#102a43,#1976a3); color:white;
            border-radius:14px; padding:2rem 2.2rem; margin-bottom:1.2rem; }
    .hero h1 { margin:0; font-size:2.2rem; letter-spacing:-.03em; }
    .hero p { margin:.5rem 0 0; color:#d9e2ec; max-width:760px; }
    .evidence { background:white; border:1px solid var(--line); border-radius:12px;
                padding:1rem 1.15rem; margin:.6rem 0; box-shadow:0 2px 8px #102a4310; }
    .evidence-label { color:var(--slate); font-size:.78rem; text-transform:uppercase;
                      letter-spacing:.08em; font-weight:700; }
    .evidence-text { color:#243b53; line-height:1.55; margin-top:.35rem; }
    .muted { color:var(--slate); font-size:.9rem; }
    .stButton > button { border-radius:8px; border-color:#9fb3c8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    return resources()


@st.cache_data
def load_classified_rows(root: Path) -> list[dict[str, str]]:
    path = root / "data" / "processed" / "classified_contexts.csv"
    return read_csv(path) if path.is_file() else []


dataset, bm25, explanations = load_resources()
classified_rows = load_classified_rows(settings.root)
stats = dataset_statistics(dataset)


def title_for_retracted(record_id: str) -> str:
    row = next(
        (row for row in dataset.retracted if row.get("retracted_record_id") == record_id),
        {},
    )
    return row.get("retracted_title", "Retracted paper not found")


def citing_for(openalex_id: str) -> dict[str, str]:
    return next(
        (row for row in dataset.citing if row.get("openalex_id") == openalex_id),
        {},
    )


def contexts_for(openalex_id: str | None = None, record_id: str | None = None):
    rows = dataset.contexts
    if openalex_id:
        rows = [row for row in rows if row.get("citing_openalex_id") == openalex_id]
    if record_id:
        rows = [row for row in rows if row.get("retracted_record_id") == record_id]
    return rows


def classification_for(context: dict[str, str]) -> dict[str, object]:
    row = next(
        (
            row for row in classified_rows
            if row.get("citing_openalex_id") == context.get("citing_openalex_id")
            and row.get("reference_id") == context.get("reference_id")
            and row.get("citation_marker") == context.get("citation_marker")
            and row.get("citation_sentence") == context.get("citation_sentence")
        ),
        None,
    )
    if row and row.get("predicted_category"):
        return {
            "category": row["predicted_category"],
            "confidence": row.get("prediction_confidence", ""),
            "method": row.get("classification_method", "rule_based"),
            "evidence": row.get("classification_evidence", ""),
            "human_label": row.get("human_label", ""),
        }
    prediction = CitationClassifier().predict(context)
    return {
        "category": prediction.category,
        "confidence": f"{prediction.confidence:.2f}",
        "method": prediction.method,
        "evidence": "; ".join(prediction.evidence),
        "human_label": "",
    }


def render_evidence(context: dict[str, str], show_paragraph: bool = True) -> None:
    citing = citing_for(context.get("citing_openalex_id", ""))
    classification = classification_for(context)
    human_label = classification["human_label"] or "Not human-validated"
    confidence = classification["confidence"] or "Not available"
    st.markdown('<div class="evidence">', unsafe_allow_html=True)
    st.markdown(
        f"**{citing.get('citing_title', 'Unknown citing publication')}**  \n"
        f"*{title_for_retracted(context.get('retracted_record_id', ''))}*",
    )
    st.markdown(
        f'<span class="muted">{citing.get("publication_date", "Date unavailable")} · '
        f'{context.get("citation_timing", "DATE_UNCERTAIN")} · '
        f'classification: {classification["category"]} · confidence: {confidence}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="evidence-label">Citation sentence</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="evidence-text">{context.get("citation_sentence", "")}</div>', unsafe_allow_html=True)
    if show_paragraph:
        st.markdown('<div class="evidence-label">Surrounding paragraph</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="evidence-text">{context.get("paragraph_text", "")}</div>', unsafe_allow_html=True)
    st.caption(
        f"Section: {context.get('section_heading') or 'Unavailable'} · "
        f"Evidence: {classification['evidence'] or 'No rule evidence'} · "
        f"Source: {context.get('xml_file') or 'Unavailable'} · "
        f"Human label: {human_label}",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def investigation_report(record: dict[str, str], contexts: list[dict[str, str]]) -> str:
    classification_counts = Counter(classification_for(row)["category"] for row in contexts)
    before = sum(row.get("citation_timing") == "BEFORE_RETRACTION" for row in contexts)
    after = sum(row.get("citation_timing") == "AFTER_RETRACTION" for row in contexts)
    lines = [
        "# RetractTrace Investigation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "Review verified citation evidence associated with a selected retracted publication.",
        "",
        "## Publication profile",
        f"- Title: {record.get('retracted_title', '')}",
        f"- DOI: {record.get('retracted_doi', '')}",
        f"- Journal: {record.get('journal', '')}",
        f"- Original publication date: {record.get('original_paper_date', '')}",
        f"- Retraction date: {record.get('retraction_date', '')}",
        f"- Retraction reason: {record.get('retraction_reasons', '')}",
        "",
        "## Citation statistics",
        f"- Extracted citation contexts: {len(contexts)}",
        f"- Before retraction: {before}",
        f"- After retraction: {after}",
        f"- Classification distribution: {dict(classification_counts)}",
        "",
        "## Evidence",
    ]
    for index, context in enumerate(contexts, start=1):
        prediction = classification_for(context)
        lines.extend(
            [
                f"### Context {index}",
                f"- Citing publication: {citing_for(context.get('citing_openalex_id', '')).get('citing_title', '')}",
                f"- Timing: {context.get('citation_timing', 'DATE_UNCERTAIN')}",
                f"- Category: {prediction['category']} ({prediction['confidence']})",
                f"- Source: {context.get('xml_file', '')}",
                "",
                f"> {context.get('citation_sentence', '')}",
                "",
                context.get("paragraph_text", ""),
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "Classification predictions are not human-validated labels. Citation and graph relationships do not prove reliance, causality, or scientific harm. Missing full text and unavailable metadata remain outside this report.",
        ]
    )
    return "\n".join(lines)


def overview() -> None:
    st.markdown(
        '<div class="hero"><h1>RetractTrace</h1><p>Evidence-led investigation of how '
        'retracted publications continue to appear in scholarly literature.</p></div>',
        unsafe_allow_html=True,
    )
    query = st.text_input("Start an investigation", placeholder="Search a paper title, DOI, or research topic")
    if query:
        st.session_state["discovery_query"] = query
        st.info("Open Paper Discovery from the navigation to review matching publications.")
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ("Retracted papers", "Citing publications", "Citation contexts", "Downloaded XML"),
        (stats["retracted_papers"], stats["citing_papers"], stats["citation_contexts"], stats["xml_downloaded"]),
    ):
        col.metric(label, value)
    left, right = st.columns(2)
    with left:
        st.subheader("Citation timing")
        st.bar_chart(
            {
                "Contexts": {
                    "Before retraction": stats["contexts_before_retraction"],
                    "After retraction": stats["contexts_after_retraction"],
                    "Date uncertain": stats["citation_contexts"] - stats["contexts_before_retraction"] - stats["contexts_after_retraction"],
                }
            }
        )
    with right:
        st.subheader("Classification distribution")
        counts = Counter(classification_for(row)["category"] for row in dataset.contexts)
        st.bar_chart({"Contexts": dict(counts)})
    st.caption(
        "Coverage is based on the verified Phase 1 outputs. Categories are rule-based "
        "predictions unless a genuine human label is present.",
    )


def discovery() -> None:
    st.title("Paper Discovery")
    default_query = st.session_state.get("discovery_query", "")
    query = st.text_input("Search title, DOI, author, journal, or topic", value=default_query)
    source = st.selectbox("Record type", ["All", "Retracted papers", "Citing publications"])
    reason = st.text_input("Retraction reason contains", placeholder="Optional")
    results: list[tuple[str, dict[str, str]]] = []
    needle = query.lower().strip()
    for row in dataset.retracted if source != "Citing publications" else []:
        haystack = " ".join(row.get(key, "") for key in ("retracted_title", "retracted_doi", "journal", "retraction_reasons")).lower()
        if (not needle or needle in haystack) and (not reason or reason.lower() in row.get("retraction_reasons", "").lower()):
            results.append(("retracted", row))
    for row in dataset.citing if source != "Retracted papers" else []:
        haystack = " ".join(row.get(key, "") for key in ("citing_title", "citing_doi", "citing_authors", "citing_journal", "citing_abstract")).lower()
        if needle and needle not in haystack:
            continue
        results.append(("citing", row))
    st.caption(f"{len(results)} matching records")
    for kind, row in results[:100]:
        title = row.get("retracted_title" if kind == "retracted" else "citing_title", "Untitled")
        with st.expander(title):
            if kind == "retracted":
                st.write(f"DOI: {row.get('retracted_doi', 'Unavailable')}")
                st.write(f"Original publication: {row.get('original_paper_date', 'Unavailable')}")
                st.write(f"Retraction date: {row.get('retraction_date', 'Unavailable')}")
                st.write(f"Reason: {row.get('retraction_reasons', 'Unavailable')}")
                if st.button("Investigate paper", key=f"investigate-{row.get('retracted_record_id')}"):
                    st.session_state["selected_record_id"] = row.get("retracted_record_id", "")
                    st.session_state["section"] = "Citation Intelligence"
                    st.rerun()
            else:
                st.write(f"DOI: {row.get('citing_doi', 'Unavailable')}")
                st.write(f"Authors: {row.get('citing_authors', 'Unavailable')}")
                st.write(f"Journal: {row.get('citing_journal', 'Unavailable')}")
                st.write(f"Publication date: {row.get('publication_date', 'Unavailable')}")


def citation_intelligence() -> None:
    st.title("Citation Intelligence")
    record_ids = sorted({row.get("retracted_record_id", "") for row in dataset.contexts})
    record_options = ["All"] + [
        f"{record_id} — {title_for_retracted(record_id)}" for record_id in record_ids
    ]
    selected_record = st.selectbox(
        "Investigated retracted paper",
        record_options,
    )
    selected_record_id = (
        ""
        if selected_record == "All"
        else selected_record.split(" — ", 1)[0]
    )
    category_options = ["All"] + list(CATEGORIES)
    category = st.selectbox("Citation category", category_options)
    timing = st.selectbox("Retraction timing", ["All", "BEFORE_RETRACTION", "AFTER_RETRACTION", "DATE_UNCERTAIN"])
    rows = contexts_for(record_id=selected_record_id or None)
    filtered = []
    for row in rows:
        prediction = classification_for(row)
        if category != "All" and prediction["category"] != category:
            continue
        if timing != "All" and row.get("citation_timing") != timing:
            continue
        filtered.append(row)
    st.caption(f"{len(filtered)} evidence records")
    if selected_record_id:
        record = next(
            row for row in dataset.retracted
            if row.get("retracted_record_id") == selected_record_id
        )
        st.info(f"Retraction reason: {record.get('retraction_reasons', 'Unavailable')}")
        st.download_button(
            "Export investigation report",
            investigation_report(record, filtered),
            file_name=f"retracttrace-{selected_record_id}.md",
            mime="text/markdown",
        )
    for row in filtered:
        render_evidence(row)


def graph_explorer() -> None:
    st.title("Knowledge Graph Explorer")
    records = {row.get("retracted_record_id", ""): row for row in dataset.retracted}
    record_options = [
        f"{record_id} — {record.get('retracted_title', record_id)}"
        for record_id, record in records.items()
    ]
    selected_option = st.selectbox("Retracted paper", record_options)
    selected = selected_option.split(" — ", 1)[0]
    record = records[selected]
    st.write(f"**Retraction reason:** {record.get('retraction_reasons', 'Unavailable')}")
    graph = GraphService()
    try:
        graph.connect()
        paper_id = record.get("openalex_id") or record.get("retracted_doi", "")
        paths = graph.paths(paper_id, hops=2)
        st.caption("Actual Neo4j paths; graph connectivity does not establish reliance or causality.")
        if paths:
            st.dataframe(paths, use_container_width=True, hide_index=True)
        else:
            st.info("No paths were returned for this paper.")
    except RuntimeError as error:
        st.error(str(error))
    finally:
        graph.close()
    st.markdown("**Relationship legend:** `RETRACTED_FOR`, `CITES`, `HAS_CONTEXT`.")


def retrieval_explainability() -> None:
    st.title("Retrieval & Explainability")
    query = st.text_input("Research question or search query", placeholder="e.g. air resistance particle movement")
    top_k = st.slider("Results per method", 1, 20, 5)
    if not query:
        st.info("Enter one query to compare all available retrieval approaches.")
        return
    bm25_results = bm25.search(query, top_k)
    try:
        dense = dense_resource()
        dense_results = dense.search(query, top_k)
        hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k)
    except (ImportError, RuntimeError) as error:
        dense_results, hybrid_results = [], []
        st.warning(f"Dense and hybrid retrieval unavailable: {error}")
    graph_results: list[dict[str, object]] = []
    graph = GraphService()
    try:
        graph.connect()
        graph_results = graph.search_papers(query, top_k)
    except RuntimeError as error:
        st.warning(f"Graph retrieval unavailable: {error}")
    finally:
        graph.close()
    tabs = st.tabs(["BM25", "Dense", "Hybrid", "Graph-enhanced"])
    result_sets = [bm25_results, dense_results, hybrid_results, graph_results]
    for tab, results, method in zip(tabs, result_sets, ("BM25", "Dense", "Hybrid", "Graph")):
        with tab:
            if not results:
                st.info("No results or method unavailable.")
            for rank, result in enumerate(results, start=1):
                if isinstance(result, dict):
                    paper = result
                    title = paper.get("title", "")
                    score = "rank-based graph score"
                    explanation = "Neo4j title/DOI text match"
                    paper_id = paper.get("id", "")
                else:
                    title = result.title
                    score = f"{result.score:.4f}"
                    explanation = " · ".join(result.explanation)
                    paper_id = result.paper_id
                st.markdown(f"**{rank}. {title}**")
                st.caption(f"{method} · score: {score} · {explanation}")
                contexts = contexts_for(paper_id)
                if contexts:
                    render_evidence(contexts[0], show_paragraph=False)


section = st.session_state.get("section", "Overview")
with st.sidebar:
    st.markdown("## RetractTrace")
    st.caption("Evidence-led scholarly investigation")
    section = st.radio(
        "Workspace",
        ["Overview", "Paper Discovery", "Citation Intelligence", "Knowledge Graph Explorer", "Retrieval & Explainability"],
        index=["Overview", "Paper Discovery", "Citation Intelligence", "Knowledge Graph Explorer", "Retrieval & Explainability"].index(section),
    )
st.session_state["section"] = section

if section == "Overview":
    overview()
elif section == "Paper Discovery":
    discovery()
elif section == "Citation Intelligence":
    citation_intelligence()
elif section == "Knowledge Graph Explorer":
    graph_explorer()
else:
    retrieval_explainability()
