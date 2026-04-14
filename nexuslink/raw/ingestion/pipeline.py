"""Top-level ingestion pipeline: route a source to the correct loader, extract entities,
and write an Obsidian-compatible Paper note into wiki/papers/.
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

# wiki/papers/ relative to this file:  raw/ingestion/../../wiki/papers
_WIKI_PAPERS_DIR = Path(__file__).parent.parent.parent / "wiki" / "papers"

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
# DOI ingestion via CrossRef (returns metadata only; no full text)
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

    return RawDocument(
        title=title,
        authors=authors,
        doi=doi,
        abstract=msg.get("abstract", ""),
        full_text=msg.get("abstract", ""),  # CrossRef doesn't give full text
        source_path=f"doi:{doi}",
        year=year,
        domain_tags=[s.get("name", "") for s in msg.get("subject", [])],
    )


# ---------------------------------------------------------------------------
# Obsidian wiki note writer
# ---------------------------------------------------------------------------

async def _write_wiki_note(doc: RawDocument, entities: list) -> Path:
    """Render a Paper.md note into wiki/papers/ using [[wikilinks]] for entities."""
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
tags: []
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
