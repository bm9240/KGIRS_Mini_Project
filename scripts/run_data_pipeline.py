"""Acquire and process citation evidence for retracted papers.

The pipeline uses only public OpenAlex and NCBI/PMC services. Network
responses are cached under data/interim so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import tarfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
RAW_RETRACTION_FILE = ROOT / "data" / "raw" / "retraction_watch" / "retraction_watch.csv"
RAW_XML_DIR = ROOT / "data" / "raw" / "pmc_xml"
INTERIM_DIR = ROOT / "data" / "interim"
OPENALEX_CACHE = INTERIM_DIR / "openalex_cache"
NCBI_CACHE = INTERIM_DIR / "ncbi_cache"
PMC_CACHE = INTERIM_DIR / "pmc_cache"
PROCESSED_DIR = ROOT / "data" / "processed"
SELECTED_FILE = PROCESSED_DIR / "selected_retracted_papers.csv"
CITING_FILE = PROCESSED_DIR / "citing_papers.csv"
CONTEXT_FILE = PROCESSED_DIR / "citation_contexts.csv"
REPORT_FILE = PROCESSED_DIR / "acquisition_report.csv"
UNAVAILABLE_FILE = INTERIM_DIR / "unavailable_articles.csv"
FAILED_MATCH_FILE = INTERIM_DIR / "failed_matches.csv"
CITING_FIELDS = [
    "openalex_id", "citing_title", "citing_doi", "citing_pmid",
    "citing_authors", "citing_journal", "citing_abstract",
    "publication_date", "retracted_record_ids", "pmcid",
    "pmc_resolution_status", "pmc_license", "xml_status", "xml_file",
]
CONTEXT_FIELDS = [
    "citing_openalex_id", "citing_doi", "citing_pmid", "citing_pmcid",
    "retracted_record_id", "retracted_doi", "retracted_pmid", "reference_id",
    "reference_text", "citation_marker", "citation_sentence", "paragraph_text",
    "section_heading", "matching_method", "matching_confidence",
    "citation_timing", "xml_file",
]
FAILURE_FIELDS = ["stage", "identifier", "status", "detail"]

LOGGER = logging.getLogger("retracttrace.pipeline")
DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
PMID_RE = re.compile(r"(?:pubmed(?:\.ncbi\.nlm\.nih\.gov/)?|pmid:?)?(\d{1,})", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
ACCEPTED_LICENSE_RE = re.compile(
    r"(?:creative\s+commons|cc\s+by|public\s+domain|^pd$|^pd\s)", re.IGNORECASE
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    ncbi_email: str
    ncbi_api_key: str
    openalex_email: str
    request_delay: float
    max_citing_per_retracted: int | None

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            ncbi_email=os.getenv("PMC_EMAIL", ""),
            ncbi_api_key=os.getenv("PMC_API_KEY", ""),
            openalex_email=os.getenv("OPENALEX_EMAIL", os.getenv("PMC_EMAIL", "")),
            request_delay=max(0.1, float(os.getenv("RETRACTTRACE_REQUEST_DELAY", "0.35"))),
            max_citing_per_retracted=_optional_int(
                os.getenv("OPENALEX_MAX_CITING_PER_RETRACTED")
            ),
        )


class JsonCache:
    """Small file cache for JSON API responses."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Ignoring invalid cache file %s", path)
            return None

    def put(self, key: str, value: Any) -> None:
        path = self.path_for(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        temporary.replace(path)


class ApiClient:
    """Rate-limited HTTP client with retries and explicit request failures."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "RetractTrace/1.0 (research data pipeline)"}
        )
        self.last_request = 0.0

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        cache: JsonCache | None = None,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        key = cache_key or _cache_key(url, params)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                return cached

        last_error = "unknown error"
        for attempt in range(4):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=60)
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", "2"))
                    time.sleep(min(60.0, retry_after))
                    last_error = "HTTP 429 rate limited"
                    continue
                if 400 <= response.status_code < 500:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                if cache is not None:
                    cache.put(key, payload)
                return payload
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code != 429:
                    raise RuntimeError(f"HTTP {error.response.status_code}: {url}") from error
                last_error = str(error)
                LOGGER.warning("Request failed (%s/4) %s: %s", attempt + 1, url, error)
            except (requests.RequestException, ValueError) as error:
                last_error = str(error)
                LOGGER.warning("Request failed (%s/4) %s: %s", attempt + 1, url, error)
                if attempt < 3:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Request failed after retries: {url}: {last_error}")

    def get_bytes(self, url: str, *, cache: JsonCache, cache_key: str) -> bytes:
        path = cache.directory / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.bin"
        if path.is_file():
            return path.read_bytes()
        last_error = "unknown error"
        for attempt in range(4):
            self._wait()
            try:
                response = self.session.get(url, timeout=120)
                if response.status_code == 429:
                    time.sleep(min(60.0, float(response.headers.get("Retry-After", "2"))))
                    continue
                if 400 <= response.status_code < 500:
                    response.raise_for_status()
                response.raise_for_status()
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(path)
                return response.content
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code != 429:
                    raise RuntimeError(f"HTTP {error.response.status_code}: {url}") from error
                last_error = str(error)
                LOGGER.warning("Download failed (%s/4) %s: %s", attempt + 1, url, error)
            except requests.RequestException as error:
                last_error = str(error)
                LOGGER.warning("Download failed (%s/4) %s: %s", attempt + 1, url, error)
                if attempt < 3:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Download failed after retries: {url}: {last_error}")

    def get_text(self, url: str, *, cache: JsonCache, cache_key: str) -> str:
        path = cache.directory / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.xml"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        last_error = "unknown error"
        for attempt in range(4):
            self._wait()
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 429:
                    time.sleep(min(60.0, float(response.headers.get("Retry-After", "2"))))
                    continue
                if 400 <= response.status_code < 500:
                    response.raise_for_status()
                response.raise_for_status()
                temporary = path.with_suffix(".tmp")
                temporary.write_text(response.text, encoding="utf-8")
                temporary.replace(path)
                return response.text
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code != 429:
                    raise RuntimeError(f"HTTP {error.response.status_code}: {url}") from error
                last_error = str(error)
                LOGGER.warning("Request failed (%s/4) %s: %s", attempt + 1, url, error)
            except requests.RequestException as error:
                last_error = str(error)
                LOGGER.warning("Request failed (%s/4) %s: %s", attempt + 1, url, error)
                if attempt < 3:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Request failed after retries: {url}: {last_error}")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.settings.request_delay:
            time.sleep(self.settings.request_delay - elapsed)
        self.last_request = time.monotonic()


def load_retracted_papers(limit: int) -> list[dict[str, str]]:
    """Load, normalize, deduplicate, and select usable retraction records."""
    if not RAW_RETRACTION_FILE.is_file():
        raise FileNotFoundError(f"Input dataset not found: {RAW_RETRACTION_FILE}")

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    with RAW_RETRACTION_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Record ID", "Title", "OriginalPaperDOI", "OriginalPaperPubMedID"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Retraction Watch CSV is missing columns: {sorted(missing)}")
        for row in reader:
            doi = normalize_doi(row.get("OriginalPaperDOI", ""))
            pmid = normalize_pmid(row.get("OriginalPaperPubMedID", ""))
            year = _publication_year(row.get("OriginalPaperDate", ""))
            key = f"doi:{doi.lower()}" if doi else f"pmid:{pmid}" if pmid else ""
            if not year or not 2018 <= year <= 2021 or not key or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "retracted_record_id": row.get("Record ID", "").strip(),
                    "retracted_title": row.get("Title", "").strip(),
                    "retracted_doi": doi,
                    "retracted_pmid": pmid,
                    "retraction_date": row.get("RetractionDate", "").strip(),
                    "original_paper_date": row.get("OriginalPaperDate", "").strip(),
                    "original_publication_year": str(year),
                    "journal": row.get("Journal", "").strip(),
                    "retraction_reasons": row.get("Reason", "").strip(),
                    "openalex_id": "",
                    "openalex_cited_by_count": "",
                    "crossref_title": "",
                    "crossref_status": "not_attempted",
                }
            )
    grouped: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["journal"] or "Unknown journal", []).append(candidate)
    selected = []
    journal_names = sorted(grouped)
    while len(selected) < limit and journal_names:
        next_names = []
        for journal in journal_names:
            if grouped[journal]:
                selected.append(grouped[journal].pop(0))
                if len(selected) == limit:
                    break
            if grouped[journal]:
                next_names.append(journal)
        journal_names = next_names
    _write_csv(SELECTED_FILE, selected)
    return selected


def discover_citing_papers(
    papers: list[dict[str, str]], client: ApiClient, settings: Settings
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Find citing works through OpenAlex and return records and failures."""
    cache = JsonCache(OPENALEX_CACHE)
    citing: dict[str, dict[str, str]] = {}
    failures: list[dict[str, str]] = []
    for paper in papers:
        identifier = paper["retracted_doi"] or paper["retracted_pmid"]
        try:
            work = _openalex_work(paper, client, cache, settings)
            _enrich_crossref(paper, client)
            openalex_id = _work_id(work)
            if not openalex_id:
                raise ValueError("OpenAlex work was not found")
            paper["openalex_id"] = cited_work_id = openalex_id.rsplit("/", 1)[-1]
            paper["openalex_cited_by_count"] = str(work.get("cited_by_count", 0))
            cited_work_id = openalex_id.rsplit("/", 1)[-1]
            cited_url = "https://api.openalex.org/works"
            cursor = "*"
            found = 0
            while cursor:
                payload = client.get_json(
                    cited_url,
                    params={
                        "filter": f"cites:{cited_work_id},is_paratext:false",
                        "per-page": "200",
                        "cursor": cursor,
                        "select": (
                            "id,doi,title,publication_date,ids,primary_location,"
                            "authorships,open_access,abstract_inverted_index"
                        ),
                    },
                    cache=cache,
                    cache_key=_cache_key(cited_url, {
                        "filter": f"cites:{cited_work_id},is_paratext:false",
                        "per-page": "200",
                        "cursor": cursor,
                    }),
                )
                for work_item in payload.get("results", []):
                    record = _citing_record(work_item, paper)
                    key = record["openalex_id"] or record["citing_doi"] or record["citing_pmid"]
                    if key and key not in citing:
                        citing[key] = record
                        found += 1
                    elif key:
                        existing = citing[key]["retracted_record_ids"].split(";")
                        if record["retracted_record_ids"] not in existing:
                            existing.append(record["retracted_record_ids"])
                            citing[key]["retracted_record_ids"] = ";".join(existing)
                    if (
                        settings.max_citing_per_retracted is not None
                        and found >= settings.max_citing_per_retracted
                    ):
                        cursor = ""
                        break
                else:
                    cursor = payload.get("meta", {}).get("next_cursor", "")
            if found == 0 and int(paper.get("openalex_cited_by_count", "0")) > 0:
                failures.append(
                    {
                        "stage": "openalex",
                        "identifier": identifier,
                        "status": "validation_warning",
                        "detail": (
                            f"OpenAlex work reports cited_by_count="
                            f"{paper['openalex_cited_by_count']} but cites query returned zero"
                        ),
                    }
                )
        except (RuntimeError, ValueError, TypeError) as error:
            failures.append(
                {
                    "stage": "openalex",
                    "identifier": identifier,
                    "status": "failed",
                    "detail": str(error),
                }
            )
            LOGGER.error("OpenAlex failed for %s: %s", identifier, error)
    records = list(citing.values())
    _write_csv(SELECTED_FILE, papers)
    _write_csv(CITING_FILE, records, CITING_FIELDS)
    return records, failures


def resolve_pmcids(
    citing: list[dict[str, str]], client: ApiClient, settings: Settings
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Resolve OpenAlex DOI/PMID identifiers to PMCIDs using NCBI ID Converter."""
    cache = JsonCache(NCBI_CACHE)
    identifier_groups = {
        "doi": sorted({record["citing_doi"] for record in citing if record["citing_doi"]}),
        "pmid": sorted({record["citing_pmid"] for record in citing if record["citing_pmid"]}),
    }
    LOGGER.info(
        "Resolving %d DOI and %d PMID identifiers to PMCIDs",
        len(identifier_groups["doi"]),
        len(identifier_groups["pmid"]),
    )
    mappings: dict[str, dict[str, str]] = {}
    failures: list[dict[str, str]] = []
    for id_type, identifiers in identifier_groups.items():
        for batch_start in range(0, len(identifiers), 100):
            batch = identifiers[batch_start : batch_start + 100]
            params = {
                "ids": ",".join(batch),
                "idtype": id_type,
                "format": "json",
                "tool": "RetractTrace",
                "email": settings.ncbi_email or "retracttrace@example.org",
            }
            if settings.ncbi_api_key:
                params["api_key"] = settings.ncbi_api_key
            try:
                LOGGER.info(
                    "Submitting NCBI %s batch %d (%d identifiers)",
                    id_type,
                    batch_start // 100 + 1,
                    len(batch),
                )
                payload = client.get_json(
                    "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                    params=params,
                    cache=cache,
                    cache_key=_cache_key("idconv", params),
                )
                for converted in payload.get("records", []):
                    for key in (converted.get("doi"), converted.get("pmid")):
                        if key:
                            mappings[str(key).lower()] = converted
            except RuntimeError as error:
                failures.append(
                    {
                        "stage": "pmc_resolution",
                        "identifier": ",".join(batch),
                        "status": "failed",
                        "detail": str(error),
                    }
                )
    for record in citing:
        match = mappings.get(record["citing_doi"].lower()) or mappings.get(
            record["citing_pmid"].lower()
        )
        record["pmcid"] = normalize_pmcid(match.get("pmcid", "")) if match else ""
        record["pmc_resolution_status"] = "identified" if record["pmcid"] else "not_found"
        record["pmc_license"] = ""
        record["xml_status"] = "not_attempted"
        record["xml_file"] = ""
        if not record["pmcid"]:
            failures.append(
                {
                    "stage": "pmc_resolution",
                    "identifier": record["openalex_id"] or record["citing_doi"],
                    "status": "unavailable",
                    "detail": "No PMCID returned by NCBI ID Converter",
                }
            )
    _write_csv(CITING_FILE, citing, CITING_FIELDS)
    return citing, failures


def acquire_xml(
    citing: list[dict[str, str]], client: ApiClient, settings: Settings
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Check PMC OA reuse licences and download eligible JATS XML packages."""
    cache = JsonCache(PMC_CACHE)
    failures: list[dict[str, str]] = []
    for record in citing:
        pmcid = record["pmcid"]
        if not pmcid:
            continue
        try:
            xml_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pmc&id={quote(pmcid)}&rettype=full&retmode=xml"
                f"&tool=RetractTrace&email={quote(settings.ncbi_email or 'retracttrace@example.org')}"
            )
            if settings.ncbi_api_key:
                xml_url += f"&api_key={quote(settings.ncbi_api_key)}"
            xml_text = client.get_text(
                xml_url,
                cache=cache,
                cache_key=f"efetch:{pmcid}",
            )
            xml_bytes = xml_text.encode("utf-8")
            etree.fromstring(xml_bytes)
            license_name = _license_from_xml(xml_bytes)
            record["pmc_license"] = license_name
            if not is_reuse_license(license_name):
                record["xml_status"] = "unavailable_license"
                failures.append(
                    {
                        "stage": "xml_acquisition",
                        "identifier": pmcid,
                        "status": "unavailable",
                        "detail": f"Missing or non-reuse PMC OA licence: {license_name or 'unknown'}",
                    }
                )
                continue
            xml_path = RAW_XML_DIR / f"{pmcid}.xml"
            if not xml_path.is_file():
                xml_path.write_bytes(xml_bytes)
            record["xml_status"] = "downloaded"
            record["xml_file"] = str(xml_path.relative_to(ROOT))
        except (RuntimeError, ValueError, tarfile.TarError, OSError) as error:
            record["xml_status"] = "failed"
            failures.append(
                {
                    "stage": "xml_acquisition",
                    "identifier": pmcid,
                    "status": "failed",
                    "detail": str(error),
                }
            )
    _write_csv(CITING_FILE, citing, CITING_FIELDS)
    return citing, failures


def extract_contexts(
    citing: list[dict[str, str]], papers: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Extract citation markers and context paragraphs and match target references."""
    targets: dict[str, dict[str, str]] = {}
    for paper in papers:
        if paper["retracted_doi"]:
            targets[f"doi:{paper['retracted_doi']}"] = paper
        if paper["retracted_pmid"]:
            targets[f"pmid:{paper['retracted_pmid']}"] = paper

    contexts: list[dict[str, str]] = []
    failed_matches: list[dict[str, str]] = []
    for record in citing:
        if record["xml_status"] != "downloaded":
            continue
        xml_path = ROOT / record["xml_file"]
        try:
            root = etree.parse(str(xml_path)).getroot()
            references = _reference_index(root)
            marker_count = 0
            matched_count = 0
            for xref in root.xpath(
                './/xref[@ref-type="bibr" or @ref-type="reference"]'
            ):
                if _ancestor(xref, "ref") is not None:
                    continue
                marker_count += 1
                rid_values = (xref.get("rid") or "").split()
                for rid in rid_values:
                    reference = references.get(rid)
                    if not reference:
                        continue
                    matched_targets = _unique_targets([
                        targets[key]
                        for key in reference["keys"]
                        if key in targets
                    ])
                    matching_method = "DOI_OR_PMID"
                    matching_confidence = "1.0"
                    if not matched_targets:
                        title_matches = [
                            (target, method)
                            for target in papers
                            for matched, method in [_target_title_matches(reference["title"], target)]
                            if matched
                        ]
                        matched_targets = _unique_targets(
                            [target for target, _ in title_matches]
                        )
                        matching_method = (
                            title_matches[0][1] if title_matches else "TITLE"
                        )
                        matching_confidence = "0.85" if matched_targets else ""
                    if not matched_targets:
                        continue
                    for target in matched_targets:
                        matched_count += 1
                        paragraph = _context_container(xref)
                        paragraph_text = _text(paragraph)
                        contexts.append(
                            {
                                "citing_openalex_id": record["openalex_id"],
                                "citing_doi": record["citing_doi"],
                                "citing_pmid": record["citing_pmid"],
                                "citing_pmcid": record["pmcid"],
                                "retracted_record_id": target["retracted_record_id"],
                                "retracted_doi": target["retracted_doi"],
                                "retracted_pmid": target["retracted_pmid"],
                                "reference_id": rid,
                                "reference_text": reference["text"],
                                "citation_marker": _text(xref),
                                "citation_sentence": _citation_sentence(paragraph, xref),
                                "paragraph_text": paragraph_text,
                                "section_heading": _section_heading(xref),
                                "matching_method": matching_method,
                                "matching_confidence": matching_confidence,
                                "citation_timing": _citation_timing(
                                    record["publication_date"],
                                    target["retraction_date"],
                                ),
                                "xml_file": record["xml_file"],
                            }
                        )
            if marker_count and not matched_count:
                failed_matches.append(
                    {
                        "stage": "citation_matching",
                        "identifier": record["pmcid"],
                        "status": "no_match",
                        "detail": f"{marker_count} citation markers but no selected retracted reference",
                    }
                )
        except (OSError, etree.XMLSyntaxError) as error:
            failed_matches.append(
                {
                    "stage": "xml_processing",
                    "identifier": record["pmcid"],
                    "status": "failed",
                    "detail": str(error),
                }
            )
    _write_csv(CONTEXT_FILE, contexts, CONTEXT_FIELDS)
    _write_csv(FAILED_MATCH_FILE, failed_matches, FAILURE_FIELDS)
    return contexts, failed_matches


def run(limit: int = 100) -> dict[str, int]:
    """Execute every Phase 1 stage and write the requested output files."""
    for directory in (RAW_XML_DIR, INTERIM_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")
    settings = Settings.from_environment()
    client = ApiClient(settings)
    papers = load_retracted_papers(limit)
    citing, openalex_failures = discover_citing_papers(papers, client, settings)
    citing, pmc_failures = resolve_pmcids(citing, client, settings)
    citing, xml_failures = acquire_xml(citing, client, settings)
    contexts, match_failures = extract_contexts(citing, papers)
    failures = openalex_failures + pmc_failures + xml_failures
    _write_csv(
        UNAVAILABLE_FILE,
        [row for row in failures if row["status"] == "unavailable"],
        FAILURE_FIELDS,
    )
    report_rows = _report_rows(papers, citing, contexts, failures + match_failures)
    _write_csv(REPORT_FILE, report_rows)
    summary = {
        "retracted_papers_processed": len(papers),
        "citing_papers_discovered": len(citing),
        "pmcids_identified": sum(bool(row["pmcid"]) for row in citing),
        "eligible_xml_articles_downloaded": sum(
            row["xml_status"] == "downloaded" for row in citing
        ),
        "references_successfully_matched": len(contexts),
        "citation_contexts_extracted": len(contexts),
        "failed_requests_and_unavailable_articles": len(failures),
    }
    for key, value in summary.items():
        LOGGER.info("%s: %s", key.replace("_", " ").capitalize(), value)
    return summary


def normalize_doi(value: Any) -> str:
    value = str(value or "").strip().strip(".,;")
    value = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    match = DOI_RE.search(value)
    return match.group(0).rstrip(".,;") if match else ""


def normalize_pmid(value: Any) -> str:
    value = str(value or "").strip()
    if not value or value in {"0", "nan", "None"}:
        return ""
    match = PMID_RE.search(value)
    return match.group(1) if match else ""


def normalize_pmcid(value: str) -> str:
    match = re.search(r"PMC\d+", value or "", re.I)
    return match.group(0).upper() if match else ""


def is_reuse_license(value: str) -> bool:
    return bool(ACCEPTED_LICENSE_RE.search(value or ""))


def _openalex_work(
    paper: dict[str, str], client: ApiClient, cache: JsonCache, settings: Settings
) -> dict[str, Any]:
    if paper["retracted_doi"]:
        identifier = f"https://doi.org/{quote(paper['retracted_doi'], safe='')}"
    else:
        identifier = f"https://pubmed.ncbi.nlm.nih.gov/{paper['retracted_pmid']}/"
    return client.get_json(
        f"https://api.openalex.org/works/{identifier}",
        params={"mailto": settings.openalex_email} if settings.openalex_email else None,
        cache=cache,
        cache_key=f"work:{identifier}",
    )


def _citing_record(work: dict[str, Any], source: dict[str, str]) -> dict[str, str]:
    ids = work.get("ids") or {}
    location = work.get("primary_location") or {}
    source_info = location.get("source") or {}
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            authors.append(author["display_name"])
    return {
        "openalex_id": _work_id(work),
        "citing_title": (work.get("title") or "").strip(),
        "citing_doi": normalize_doi(work.get("doi", "")),
        "citing_pmid": normalize_pmid(ids.get("pmid", "")),
        "citing_authors": "; ".join(authors),
        "citing_journal": str(source_info.get("display_name") or ""),
        "citing_abstract": _abstract_from_inverted_index(
            work.get("abstract_inverted_index")
        ),
        "publication_date": str(work.get("publication_date") or ""),
        "retracted_record_ids": source["retracted_record_id"],
        "pmcid": "",
        "pmc_resolution_status": "not_attempted",
        "pmc_license": "",
        "xml_status": "not_attempted",
        "xml_file": "",
    }


def _work_id(work: dict[str, Any]) -> str:
    return str(work.get("id") or "").strip()


def _enrich_crossref(paper: dict[str, str], client: ApiClient) -> None:
    """Verify selected DOI metadata through Crossref without replacing source data."""
    doi = paper["retracted_doi"]
    if not doi:
        paper["crossref_status"] = "no_doi"
        return
    try:
        payload = client.get_json(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
            cache=JsonCache(INTERIM_DIR / "crossref_cache"),
            cache_key=f"crossref:{doi.lower()}",
        )
        item = payload.get("message") or {}
        paper["crossref_title"] = " ".join(item.get("title") or [])
        paper["crossref_status"] = "verified"
    except (RuntimeError, ValueError, TypeError) as error:
        paper["crossref_status"] = f"failed:{error}"


def _abstract_from_inverted_index(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if isinstance(positions, list):
            words.extend((int(position), word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def _normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _title_matches(reference_title: str, target_title: str) -> bool:
    left = set(_normalize_title(reference_title).split())
    right = set(_normalize_title(target_title).split())
    if len(left) < 4 or len(right) < 4:
        return False
    overlap = len(left & right) / min(len(left), len(right))
    return overlap >= 0.8


def _target_title_matches(reference_title: str, target: dict[str, str]) -> tuple[bool, str]:
    if target.get("crossref_title") and _title_matches(
        reference_title, target["crossref_title"]
    ):
        return True, "TITLE_CROSSREF"
    return _title_matches(reference_title, target["retracted_title"]), "TITLE"


def _unique_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for target in targets:
        unique[target["retracted_record_id"]] = target
    return list(unique.values())


def _citation_timing(citing_date: str, retraction_date: str) -> str:
    citing = _parse_date(citing_date)
    retraction = _parse_date(retraction_date)
    if not citing or not retraction:
        return "DATE_UNCERTAIN"
    return "BEFORE_RETRACTION" if citing < retraction else "AFTER_RETRACTION"


def _parse_date(value: str) -> datetime | None:
    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", value or "")
    if match:
        month, day, year = (int(part) for part in match.groups())
        return datetime(year, month, day, tzinfo=timezone.utc)
    match = re.match(r"(\d{4})(?:-(\d{2})-(\d{2}))?", value or "")
    if match:
        year, month, day = int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1)
        return datetime(year, month, day, tzinfo=timezone.utc)
    return None


def _publication_year(value: str) -> int | None:
    parsed = _parse_date(value)
    return parsed.year if parsed else None


def _oa_record(payload: str, pmcid: str) -> dict[str, str]:
    root = etree.fromstring(payload.encode("utf-8"))
    record = root.xpath(f".//record[@id='{pmcid}']")
    if not record:
        error = _text(root.xpath(".//error")[0]) if root.xpath(".//error") else "record not found"
        raise ValueError(f"PMC OA service returned no record: {error}")
    node = record[0]
    links = node.xpath("./link[@format='tgz']")
    return {
        "license": node.get("license", "").strip(),
        "tgz_url": links[0].get("href", "").strip() if links else "",
    }


def _license_from_xml(xml_bytes: bytes) -> str:
    root = etree.fromstring(xml_bytes)
    licenses = root.xpath(
        "//*[local-name()='license' or local-name()='license-p']"
    )
    return " ".join(_text(node) for node in licenses).strip()


def _xml_from_package(package: bytes) -> bytes:
    with tarfile.open(fileobj=__import__("io").BytesIO(package), mode="r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith((".xml", ".nxml"))
        ]
        if not members:
            raise ValueError("PMC OA package contains no XML article")
        member = min(members, key=lambda item: len(Path(item.name).parts))
        if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
            raise ValueError("Unsafe path in PMC OA package")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("Unable to read XML member from PMC OA package")
        return extracted.read()


def _reference_index(root: etree._Element) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for reference in root.xpath(".//ref[@id]"):
        keys: set[str] = set()
        for pub_id in reference.xpath(".//pub-id"):
            kind = (pub_id.get("pub-id-type") or "").lower()
            value = _text(pub_id)
            if kind == "doi" and normalize_doi(value):
                keys.add(f"doi:{normalize_doi(value)}")
            elif kind in {"pmid", "pmc"} and normalize_pmid(value):
                keys.add(f"pmid:{normalize_pmid(value)}")
        title_nodes = reference.xpath(".//article-title | .//chapter-title | .//source")
        index[reference.get("id", "")] = {
            "keys": keys,
            "text": _text(reference),
            "title": _text(title_nodes[0]) if title_nodes else "",
        }
    return index


def _ancestor(element: etree._Element, name: str) -> etree._Element | None:
    for ancestor in element.iterancestors():
        if etree.QName(ancestor).localname == name:
            return ancestor
    return None


def _context_container(element: etree._Element) -> etree._Element:
    for name in ("p", "list-item", "td", "caption", "sec", "body"):
        container = _ancestor(element, name)
        if container is not None:
            return container
    return element


def _section_heading(element: etree._Element) -> str:
    section = _ancestor(element, "sec")
    if section is None:
        return ""
    titles = section.xpath("./title")
    return _text(titles[0]) if titles else ""


def _citation_sentence(paragraph: etree._Element, marker: etree._Element) -> str:
    text = _text(paragraph)
    marker_text = _text(marker)
    sentences = SENTENCE_RE.split(text)
    for sentence in sentences:
        if marker_text and marker_text in sentence:
            return sentence.strip()
    return text


def _text(element: etree._Element) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _cache_key(url: str, params: dict[str, str] | None = None) -> str:
    encoded = "&".join(f"{key}={params[key]}" for key in sorted(params or {}))
    return f"{url}?{encoded}"


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _date_sort_key(value: str) -> tuple[int, str]:
    normalized = value.strip()
    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", normalized)
    if match:
        month, day, year = match.groups()
        return (int(f"{year}{int(month):02d}{int(day):02d}"), normalized)
    match = re.match(r"(\d{4})", normalized)
    return (int(match.group(1) + "0101") if match else 99999999, normalized)


def _write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(fieldnames)
        else:
            path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report_rows(
    papers: list[dict[str, str]],
    citing: list[dict[str, str]],
    contexts: list[dict[str, str]],
    failures: list[dict[str, str]],
) -> list[dict[str, str]]:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "stage": "selection",
            "identifier": paper["retracted_doi"] or paper["retracted_pmid"],
            "status": "selected",
            "detail": paper["retracted_record_id"],
            "timestamp": now,
        }
        for paper in papers
    ]
    rows.extend(
        {
            "stage": "summary",
            "identifier": "",
            "status": key,
            "detail": str(value),
            "timestamp": now,
        }
        for key, value in {
            "citing_papers_discovered": len(citing),
            "pmcids_identified": sum(bool(row["pmcid"]) for row in citing),
            "eligible_xml_articles_downloaded": sum(
                row["xml_status"] == "downloaded" for row in citing
            ),
            "references_successfully_matched": len(contexts),
        }.items()
    )
    rows.extend({**failure, "timestamp": now} for failure in failures)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        run(args.limit)
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Pipeline stopped: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
