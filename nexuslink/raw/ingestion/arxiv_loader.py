"""ArXiv ingestion: fetches paper metadata and PDF via the arxiv Python client."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import arxiv
from loguru import logger

from ..schemas.models import RawDocument
from .pdf_loader import ingest_pdf


async def ingest_arxiv(arxiv_id: str) -> RawDocument:
    """Fetch an ArXiv paper by ID, download its PDF, and return a :class:`RawDocument`.

    Metadata from ArXiv (title, authors, abstract, categories) takes precedence
    over the PDF-extracted heuristics.
    """
    logger.info("Fetching ArXiv paper: {}", arxiv_id)

    paper = await asyncio.to_thread(_fetch_paper, arxiv_id)
    logger.debug("ArXiv metadata retrieved: {!r}", paper.title)

    with tempfile.TemporaryDirectory() as tmp_dir:
        safe_name = arxiv_id.replace("/", "_") + ".pdf"
        await asyncio.to_thread(
            paper.download_pdf, dirpath=tmp_dir, filename=safe_name
        )
        pdf_path = Path(tmp_dir) / safe_name
        doc = await ingest_pdf(pdf_path)

    # Overwrite heuristic metadata with authoritative ArXiv fields
    doc.title = paper.title
    doc.authors = [str(a) for a in paper.authors]
    doc.abstract = paper.summary
    doc.arxiv_id = arxiv_id
    doc.domain_tags = list(paper.categories)
    doc.source_path = f"arxiv:{arxiv_id}"
    if paper.published:
        doc.year = paper.published.year

    logger.info("ArXiv ingestion complete: {!r} ({} authors)", doc.title, len(doc.authors))
    return doc


def _fetch_paper(arxiv_id: str) -> arxiv.Result:
    """Blocking call to the ArXiv API — run via asyncio.to_thread."""
    client = arxiv.Client()
    results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
    if not results:
        raise ValueError(f"No ArXiv paper found for ID: {arxiv_id!r}")
    return results[0]
