"""Ingestion routes: upload PDFs, submit ArXiv IDs / DOIs, list papers."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from nexuslink.api.deps import get_nexuslink
from nexuslink.main import NexusLink

router = APIRouter(tags=["ingest"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class IngestResult(BaseModel):
    doc_id: str
    title: str
    entities_found: int
    domain_tags: list[str] = Field(default_factory=list)
    wiki_note: str = ""


class BatchIngestRequest(BaseModel):
    sources: list[str] = Field(..., min_length=1)


class BatchIngestResult(BaseModel):
    results: list[IngestResult]
    failed: list[str]
    total_ingested: int


class PaperSummary(BaseModel):
    id: str
    title: str
    domain: list[str]
    entity_count: int
    authors: list[str] = Field(default_factory=list)
    year: int | None = None


class PaginatedPapers(BaseModel):
    total: int
    page: int
    page_size: int
    papers: list[PaperSummary]


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResult)
async def ingest_paper(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
    file: Annotated[UploadFile | None, File()] = None,
    arxiv_id: Annotated[str | None, Form()] = None,
    doi: Annotated[str | None, Form()] = None,
) -> IngestResult:
    """Ingest a single paper.

    Send multipart/form-data with **one** of:
    - ``file`` — a PDF upload
    - ``arxiv_id`` — e.g. ``2101.03961``
    - ``doi`` — e.g. ``10.1038/nature12345``
    """
    if file is not None:
        result = await _ingest_upload(nx, file)
    elif arxiv_id:
        result = await nx.ingest(arxiv_id.strip())
    elif doi:
        result = await nx.ingest(doi.strip())
    else:
        raise HTTPException(status_code=400, detail="Provide a PDF file, arxiv_id, or doi.")

    return IngestResult(
        doc_id=result["id"],
        title=result["title"],
        entities_found=result["entities_found"],
        domain_tags=result.get("domain_tags", []),
        wiki_note=result.get("wiki_note", ""),
    )


# ---------------------------------------------------------------------------
# POST /api/ingest/batch
# ---------------------------------------------------------------------------

@router.post("/ingest/batch", response_model=BatchIngestResult)
async def ingest_batch(
    body: BatchIngestRequest,
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
) -> BatchIngestResult:
    """Ingest multiple sources concurrently (ArXiv IDs, DOIs, or file paths)."""
    raw = await asyncio.gather(
        *[nx.ingest(src) for src in body.sources],
        return_exceptions=True,
    )

    results: list[IngestResult] = []
    failed: list[str] = []

    for src, outcome in zip(body.sources, raw):
        if isinstance(outcome, Exception):
            logger.warning("Batch ingest failed for {!r}: {}", src, outcome)
            failed.append(src)
        else:
            results.append(IngestResult(
                doc_id=outcome["id"],
                title=outcome["title"],
                entities_found=outcome["entities_found"],
                domain_tags=outcome.get("domain_tags", []),
                wiki_note=outcome.get("wiki_note", ""),
            ))

    return BatchIngestResult(
        results=results,
        failed=failed,
        total_ingested=len(results),
    )


# ---------------------------------------------------------------------------
# GET /api/papers
# ---------------------------------------------------------------------------

@router.get("/papers", response_model=PaginatedPapers)
async def list_papers(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
    page: int = 1,
    page_size: int = 20,
) -> PaginatedPapers:
    """List all ingested papers from the vault, paginated."""
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be ≥ 1")
    if not (1 <= page_size <= 200):
        raise HTTPException(status_code=400, detail="page_size must be 1–200")

    papers_dir = nx._vault / "01-papers"
    md_files = sorted(papers_dir.glob("*.md")) if papers_dir.exists() else []

    summaries = await asyncio.gather(*[_parse_paper_summary(f) for f in md_files])

    total = len(summaries)
    start = (page - 1) * page_size
    page_items = list(summaries)[start : start + page_size]

    return PaginatedPapers(
        total=total,
        page=page,
        page_size=page_size,
        papers=page_items,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ingest_upload(nx: NexusLink, upload: UploadFile) -> dict:
    """Save *upload* to a temp file, call ingest, then clean up."""
    suffix = Path(upload.filename or "upload.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await upload.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return await nx.ingest(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


_WIKILINK_RE = re.compile(r"-\s+\[\[([^\]]+)\]\]")
_ENTITY_LINE_RE = re.compile(r"-\s+\[\[[^\]]+\]\]\s+\([^)]+\)")


async def _parse_paper_summary(path: Path) -> PaperSummary:
    content = await asyncio.to_thread(path.read_text, "utf-8")
    fm, body = _split_frontmatter(content)

    title: str = fm.get("title") or path.stem
    domain = _coerce_list(fm.get("domain"))
    authors = _coerce_list(fm.get("authors"))
    year_raw = fm.get("year")
    year = int(year_raw) if year_raw else None

    # Count entity wikilinks in ## Entities section
    entity_count = 0
    section = re.search(r"##\s+Entities\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL)
    if section:
        entity_count = len(_ENTITY_LINE_RE.findall(section.group(1)))

    return PaperSummary(
        id=re.sub(r"\W+", "_", title)[:64],
        title=title,
        domain=domain,
        entity_count=entity_count,
        authors=authors,
        year=year,
    )


def _split_frontmatter(content: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not m:
        return {}, content
    try:
        fm: dict = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, content[m.end():]


def _coerce_list(val: object) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str) and val:
        return [val]
    return []
