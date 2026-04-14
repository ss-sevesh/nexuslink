"""PDF ingestion: converts a PDF file to a RawDocument via pymupdf4llm."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime
from pathlib import Path

import pymupdf4llm
from loguru import logger

from ..schemas.models import RawDocument


async def ingest_pdf(path: Path) -> RawDocument:
    """Convert a PDF at *path* to a :class:`RawDocument`.

    The conversion is CPU-bound (pymupdf4llm), so it runs in a thread pool to
    avoid blocking the event loop.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    logger.info("Ingesting PDF: {}", path)
    md_text: str = await asyncio.to_thread(pymupdf4llm.to_markdown, str(path))

    title = _extract_title(md_text, path)
    authors = _extract_authors(md_text)

    logger.debug("Extracted title={!r}, authors={}", title, authors)

    return RawDocument(
        id=str(uuid.uuid4()),
        title=title,
        authors=authors,
        full_text=md_text,
        source_path=str(path),
        ingested_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Heuristic metadata extraction
# ---------------------------------------------------------------------------

def _extract_title(md: str, path: Path) -> str:
    """Return the most likely title from the converted Markdown."""
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()

    # Fallback: first substantial non-URL line
    for line in md.splitlines():
        stripped = line.strip()
        if len(stripped) > 15 and not stripped.startswith(("http", "doi:", "arXiv", "|")):
            return stripped[:200]

    return path.stem


def _extract_authors(md: str) -> list[str]:
    """Heuristically extract author names from the first 20 non-empty lines."""
    lines = [l.strip() for l in md.splitlines() if l.strip()]

    for line in lines[1 : min(20, len(lines))]:
        if line.startswith(("#", "**", "*", "-", "|", ">")):
            continue
        if any(
            kw in line.lower()
            for kw in ("abstract", "introduction", "university", "department",
                       "http", "@", "doi", "arxiv", "keyword", "©")
        ):
            continue

        # Expect comma / "and" separated short names
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", line, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if 2 <= len(parts) <= 20 and all(1 <= len(p.split()) <= 5 for p in parts):
            return parts

    return []
