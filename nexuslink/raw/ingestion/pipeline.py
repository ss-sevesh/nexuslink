"""Top-level ingestion pipeline: route a source to the correct loader, extract entities,
and write an Obsidian-compatible Paper note into wiki/01-papers/.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONTRIBUTION_RE = re.compile(
    r"\b(we propose|we present|we introduce|we show|we demonstrate|"
    r"we achieve|we find|we develop|we design|we train|we evaluate|"
    r"our model|our approach|our method|our framework|our system|our work|"
    r"this paper|this work|in this paper|in this work|"
    r"we report|we establish|we advance|we improve|we outperform)\b",
    re.IGNORECASE,
)

import httpx
from loguru import logger

from ..extraction.entity_extractor import extract_entities
from ..schemas.models import RawDocument
from .arxiv_loader import ingest_arxiv
from .pdf_loader import ingest_pdf

_JATS_TAG_RE = re.compile(r"<jats:[^>]+>|</jats:[^>]+>|<[^>]+>")
_CITATION_NUM_RE = re.compile(r"\s*\d+[–-]\d+|\s*\d+")  # inline citation superscripts


def _strip_jats_xml(text: str) -> str:
    """Remove JATS/HTML tags and inline citation numbers from CrossRef abstracts."""
    if not text:
        return ""
    text = _JATS_TAG_RE.sub("", text)
    # Collapse whitespace left by removed tags
    text = re.sub(r" {2,}", " ", text).strip()
    return text


# wiki/01-papers/ relative to this file:  raw/ingestion/../../wiki/01-papers
_WIKI_PAPERS_DIR = Path(__file__).parent.parent.parent / "wiki" / "01-papers"

_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_DOI_RE = re.compile(r"^10\.\d{4,}/.+")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_ingestion(source: str) -> dict:
    """Ingest a paper from *source* (PDF path, ArXiv ID, or DOI).

    Returns a summary dict with keys: id, title, source_type, entities_found,
    wiki_note.
    """
    source_type = _detect_source_type(source)
    logger.info("Ingesting source={!r} type={}", source, source_type)

    if source_type == "pdf":
        doc = await ingest_pdf(Path(source))
    elif source_type == "arxiv":
        doc = await ingest_arxiv(source)
    elif source_type == "doi":
        doc = await _ingest_doi(source)
    else:
        raise ValueError(f"Cannot determine source type for: {source!r}")

    entities = extract_entities(doc)
    # For DOI/abstract-only papers, entity context sentences are often empty.
    # Inject context from the abstract so embeddings carry semantic signal.
    if source_type == "doi":
        entities = _inject_abstract_context(entities, doc.abstract or "")
    note_path = await _write_wiki_note(doc, entities)

    summary = {
        "id": doc.id,
        "title": doc.title,
        "source_type": source_type,
        "entities_found": len(entities),
        "wiki_note": str(note_path.relative_to(note_path.parent.parent.parent)),
    }
    logger.info("Ingestion complete: {}", summary)
    return summary


# ---------------------------------------------------------------------------
# Source type detection
# ---------------------------------------------------------------------------

def _detect_source_type(source: str) -> str:
    """Return 'pdf', 'arxiv', or 'doi'."""
    # Explicit file path or existing file
    p = Path(source)
    if p.suffix.lower() == ".pdf" or (p.exists() and p.is_file()):
        return "pdf"

    # doi.org URL
    if "doi.org/" in source:
        doi = source.split("doi.org/")[-1]
        if _DOI_RE.match(doi):
            return "doi"

    # Bare DOI
    if _DOI_RE.match(source):
        return "doi"

    # ArXiv ID  e.g. 2101.03961 or 2101.03961v2
    if _ARXIV_RE.match(source):
        return "arxiv"

    raise ValueError(
        f"Cannot detect source type for {source!r}. "
        "Expected: path/to/file.pdf, an ArXiv ID (YYMM.NNNNN), "
        "or a DOI (10.XXXX/...)."
    )


# ---------------------------------------------------------------------------
# Unpaywall — find open-access PDF URL for a DOI
# ---------------------------------------------------------------------------

async def _unpaywall_pdf_url(doi: str) -> str | None:
    """Query Unpaywall for an open-access PDF URL.

    Returns the best OA PDF URL or None if not found / request fails.
    Unpaywall is free and requires only an email address in the query.
    """
    url = f"https://api.unpaywall.org/v2/{doi}?email=nexuslink@research.ai"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
        data = resp.json()
        # Prefer the best_oa_location PDF URL
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if pdf_url and pdf_url.endswith(".pdf"):
            return pdf_url
        # Fall back to any oa_location with a PDF
        for loc in data.get("oa_locations", []):
            u = loc.get("url_for_pdf") or loc.get("url", "")
            if u.endswith(".pdf"):
                return u
    except Exception as exc:
        logger.debug("Unpaywall lookup failed for {}: {}", doi, exc)
    return None


async def _download_tmp_pdf(url: str) -> Path | None:
    """Download a PDF from *url* to a temp file and return its path."""
    import tempfile
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(resp.content)
            return Path(f.name)
    except Exception as exc:
        logger.debug("PDF download failed from {}: {}", url, exc)
    return None


# ---------------------------------------------------------------------------
# DOI ingestion via CrossRef + Unpaywall fallback for full text
# ---------------------------------------------------------------------------

async def _ingest_doi(doi: str) -> RawDocument:
    """Fetch CrossRef metadata for *doi* and return a partial RawDocument."""
    # Normalise: strip doi.org prefix if present
    if "doi.org/" in doi:
        doi = doi.split("doi.org/")[-1]

    url = f"https://api.crossref.org/works/{doi}"
    logger.info("Resolving DOI via CrossRef: {}", doi)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers={"User-Agent": "NexusLink/0.1 (mailto:contact@nexuslink)"})
        resp.raise_for_status()

    msg = resp.json()["message"]

    title = " ".join(msg.get("title", ["Unknown Title"]))
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in msg.get("author", [])
    ]
    year_parts = msg.get("published", {}).get("date-parts", [[]])
    year = year_parts[0][0] if year_parts and year_parts[0] else None

    raw_abstract = msg.get("abstract", "")
    abstract = _strip_jats_xml(raw_abstract)

    crossref_subjects = [s.get("name", "") for s in msg.get("subject", []) if s.get("name")]

    doc = RawDocument(
        title=title,
        authors=authors,
        doi=doi,
        abstract=abstract,
        full_text=abstract,  # will be upgraded to full text if Unpaywall finds a PDF
        source_path=f"doi:{doi}",
        year=year,
        domain_tags=crossref_subjects,
    )

    # Try Unpaywall to get a full-text PDF — gives 10-50x more entities than abstract only.
    pdf_url = await _unpaywall_pdf_url(doi)
    if pdf_url:
        logger.info("Unpaywall found OA PDF for {}: {}", doi, pdf_url)
        tmp_pdf = await _download_tmp_pdf(pdf_url)
        if tmp_pdf:
            try:
                full_doc = await ingest_pdf(tmp_pdf)
                # Merge: keep CrossRef metadata, upgrade to full text + entities
                doc = RawDocument(
                    title=doc.title,
                    authors=doc.authors,
                    doi=doc.doi,
                    abstract=doc.abstract,
                    full_text=full_doc.full_text,
                    source_path=f"doi:{doi}",
                    year=doc.year,
                    domain_tags=doc.domain_tags,
                    arxiv_id=full_doc.arxiv_id,
                )
                logger.info("Upgraded DOI {} to full text ({} chars)", doi, len(doc.full_text))
            except Exception as exc:
                logger.warning("Full-text upgrade failed for {}: {}", doi, exc)
            finally:
                try:
                    tmp_pdf.unlink()
                except Exception:
                    pass

    # CrossRef subject fields are often empty for science journals (Nature, Science, PRL).
    # Fall back to keyword-based domain classification so DOI papers are never orphaned
    # into the catch-all "unknown" domain, which would prevent cross-domain bridging.
    if not doc.domain_tags:
        try:  # noqa: PLC0415
            from nexuslink.wiki.taxonomy.classifier import classify_domain
        except ImportError:
            from wiki.taxonomy.classifier import classify_domain
        ranked = classify_domain(doc)
        if ranked:
            # Take the top domain (and second if score is close) as domain tags
            top_score = ranked[0][1]
            doc.domain_tags = [d for d, s in ranked if s >= top_score * 0.6][:2]
            logger.info("DOI {} has no CrossRef subjects — classified as: {}", doi, doc.domain_tags)

    return doc


# ---------------------------------------------------------------------------
# Obsidian wiki note writer
# ---------------------------------------------------------------------------

async def _write_wiki_note(doc: RawDocument, entities: list) -> Path:
    """Render a Paper.md note into wiki/01-papers/ using [[wikilinks]] for entities."""
    import asyncio

    _WIKI_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    note_path = _WIKI_PAPERS_DIR / f"{_sanitize_filename(doc.title)}.md"
    content = _render_paper_note(doc, entities)

    await asyncio.to_thread(note_path.write_text, content, "utf-8")
    logger.info("Wrote wiki note: {}", note_path)
    return note_path


def _render_paper_note(doc: RawDocument, entities: list) -> str:
    authors_yaml = _yaml_list(doc.authors)
    domain_yaml = _yaml_list(doc.domain_tags)

    entity_lines = "\n".join(
        (
            f"- [[{e.name}]] ({e.entity_type})<!-- cx: {_trunc(e.context_sentence)} -->"
            if e.context_sentence
            else f"- [[{e.name}]] ({e.entity_type})"
        )
        for e in entities
    ) or "<!-- none detected -->"

    doi_val = doc.doi or ""
    year_val = str(doc.year) if doc.year else ""
    arxiv_val = doc.arxiv_id or ""

    # tags = domain slugs so Obsidian tag search works (#cs, #biology etc.)
    tag_slugs = [re.sub(r"[^a-z0-9_-]", "-", d.lower().strip()) for d in doc.domain_tags if d]
    tags_yaml = "[" + ", ".join(f'"{t}"' for t in tag_slugs) + "]" if tag_slugs else "[]"

    key_findings = _extract_key_findings(doc.abstract or "")
    methods_md = _extract_methods_md(entities)

    return f"""\
---
title: "{_escape_yaml(doc.title)}"
authors: {authors_yaml}
doi: "{doi_val}"
arxiv_id: "{arxiv_val}"
domain: {domain_yaml}
year: {year_val}
tags: {tags_yaml}
---

## Summary

{doc.abstract or ""}

## Key Findings

{key_findings}

## Methods

{methods_md}

## Entities

{entity_lines}

## References

"""


def _extract_key_findings(abstract: str) -> str:
    """Extract contribution sentences from an abstract using marker phrases.

    Works across domains: looks for sentences where the authors describe
    what they propose, show, achieve, or demonstrate — language common to
    NLP, biology, chemistry, physics, and materials science papers alike.
    """
    if not abstract:
        return ""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    findings = []
    for s in sentences:
        s = s.strip()
        if len(s) < 30:
            continue
        if _CONTRIBUTION_RE.search(s):
            findings.append(f"- {s}")
    return "\n".join(findings[:6])  # cap at 6 to avoid padding


def _extract_methods_md(entities: list) -> str:
    """List method-type entities as wikilinks for the ## Methods section."""
    method_entities = [e for e in entities if e.entity_type == "method"]
    if not method_entities:
        return ""
    return "\n".join(f"- [[{e.name}]]" for e in method_entities)


def _inject_abstract_context(entities: list, abstract: str) -> list:
    """Fill empty context_sentence fields by finding the abstract sentence
    that best contains or relates to each entity name.

    This ensures DOI-only papers (abstract text only) produce rich embeddings
    instead of bare entity names, enabling cross-domain bridge detection.
    """
    if not abstract or not entities:
        return entities
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 20]
    if not sentences:
        return entities

    result = []
    for e in entities:
        if e.context_sentence:
            result.append(e)
            continue
        # Find the sentence most likely to contain the entity
        name_lower = e.name.lower()
        best = next(
            (s for s in sentences if name_lower in s.lower()),
            sentences[0],  # fall back to first abstract sentence
        )
        # Pydantic models are immutable — create a copy with the new field
        result.append(e.model_copy(update={"context_sentence": best}))
    return result


def _trunc(s: str, n: int = 200) -> str:
    """Truncate a context sentence to at most n chars for inline storage.

    Newlines are collapsed to spaces so the HTML comment stays on one line —
    multi-line cx comments would break the linker's section boundary regex.
    """
    s = s.replace("-->", "—")          # escape HTML comment close tag
    s = re.sub(r"\s+", " ", s).strip() # collapse all whitespace (incl. newlines)
    return s[:n].rstrip() + ("…" if len(s) > n else "")


def _sanitize_filename(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s)
    s = s.strip(". ")
    return s[:200] or "untitled"


def _escape_yaml(s: str) -> str:
    return s.replace('"', '\\"')


def _yaml_list(items: list[str]) -> str:
    if not items:
        return "[]"
    inner = ", ".join(f'"{_escape_yaml(i)}"' for i in items)
    return f"[{inner}]"
