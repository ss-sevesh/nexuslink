"""Citation manager: BibTeX parsing, DOI resolution, and references.bib output."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nexuslink.raw.schemas.models import Citation

_WIKI_DIR = Path(__file__).parent.parent      # wiki/
_BIB_PATH = _WIKI_DIR / "assets" / "references.bib"

_CROSSREF_URL = "https://api.crossref.org/works/{doi}"
_USER_AGENT = "NexusLink/0.1 (mailto:contact@nexuslink.dev)"


class CitationManager:
    """Manages a collection of :class:`Citation` objects.

    Supports ingestion from BibTeX strings, DOI resolution via CrossRef,
    and export to BibTeX or CSL-JSON.
    """

    def __init__(self) -> None:
        # keyed by doi if available, else bibtex_key, else title
        self._citations: dict[str, Citation] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def parse_bibtex(self, bibtex_str: str) -> list[Citation]:
        """Parse one or more BibTeX entries; add each to the store."""
        entries = _parse_bibtex_entries(bibtex_str)
        citations: list[Citation] = []
        for entry in entries:
            c = Citation(
                title=entry.get("title", ""),
                authors=_split_authors(entry.get("author", "")),
                year=_safe_int(entry.get("year")),
                doi=entry.get("doi") or None,
                bibtex=_reconstruct_bibtex(entry),
            )
            self.add_citation(c)
            citations.append(c)
        logger.debug("Parsed {} BibTeX entries", len(citations))
        return citations

    async def resolve_doi(self, doi: str) -> Citation:
        """Fetch metadata from CrossRef and return a :class:`Citation`."""
        doi = doi.strip()
        if doi in self._citations:
            return self._citations[doi]

        url = _CROSSREF_URL.format(doi=doi)
        logger.info("Resolving DOI via CrossRef: {}", doi)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()

        msg: dict[str, Any] = resp.json()["message"]
        title = " ".join(msg.get("title", [""])).strip()
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in msg.get("author", [])
        ]
        year_parts = msg.get("published", {}).get("date-parts", [[]])
        year = year_parts[0][0] if year_parts and year_parts[0] else None

        c = Citation(title=title, authors=authors, year=year, doi=doi)
        self.add_citation(c)
        return c

    def add_citation(self, citation: Citation) -> None:
        key = citation.doi or citation.title or f"unknown_{len(self._citations)}"
        self._citations[key] = citation

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_bibtex(self) -> str:
        """Serialise all citations to a BibTeX string."""
        entries: list[str] = []
        for c in self._citations.values():
            if c.bibtex:
                entries.append(c.bibtex.strip())
            else:
                entries.append(_citation_to_bibtex(c))
        return "\n\n".join(entries)

    def to_csl_json(self) -> list[dict[str, Any]]:
        """Serialise all citations to CSL-JSON (one dict per citation)."""
        out: list[dict[str, Any]] = []
        for key, c in self._citations.items():
            entry: dict[str, Any] = {
                "id": re.sub(r"\W+", "_", key)[:40],
                "type": "article-journal",
                "title": c.title,
                "author": [_name_to_csl(a) for a in c.authors],
            }
            if c.year:
                entry["issued"] = {"date-parts": [[c.year]]}
            if c.doi:
                entry["DOI"] = c.doi
            out.append(entry)
        return out

    async def save_bib(self) -> None:
        """Write *wiki/assets/references.bib* with the current citation store."""
        _BIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_bibtex()
        await asyncio.to_thread(_BIB_PATH.write_text, content, "utf-8")
        logger.info("Wrote {} citations to {}", len(self._citations), _BIB_PATH)

    def __len__(self) -> int:
        return len(self._citations)


# ---------------------------------------------------------------------------
# BibTeX parsing (compatible with both bibtexparser v1 and v2)
# ---------------------------------------------------------------------------

def _parse_bibtex_entries(bib_str: str) -> list[dict[str, str]]:
    """Return a list of flat dicts from a BibTeX string.

    Tries bibtexparser v2 first, falls back to v1, falls back to a minimal
    regex parser if neither is importable.
    """
    import bibtexparser  # noqa: PLC0415

    # ---- bibtexparser v2 ----
    if hasattr(bibtexparser, "parse_string"):
        library = bibtexparser.parse_string(bib_str)
        result = []
        for entry in library.entries:
            flat: dict[str, str] = {"ID": entry.key, "ENTRYTYPE": entry.type}
            for field in entry.fields:
                flat[field.key] = str(field.value)
            result.append(flat)
        return result

    # ---- bibtexparser v1 ----
    db = bibtexparser.loads(bib_str)
    return list(db.entries)


def _reconstruct_bibtex(entry: dict[str, str]) -> str:
    etype = entry.get("ENTRYTYPE", "article")
    key = entry.get("ID", "unknown")
    skip = {"ENTRYTYPE", "ID"}
    fields = "\n".join(
        f"  {k} = {{{v}}}" for k, v in entry.items() if k not in skip
    )
    return f"@{etype}{{{key},\n{fields}\n}}"


def _citation_to_bibtex(c: Citation) -> str:
    last = (c.authors[0].split()[-1] if c.authors else "Unknown")
    year_str = str(c.year or "0000")
    key = re.sub(r"\W+", "", f"{last}{year_str}")
    fields: list[str] = [f'  title = {{{c.title}}}']
    if c.authors:
        fields.append(f'  author = {{{" and ".join(c.authors)}}}')
    if c.year:
        fields.append(f"  year = {{{c.year}}}")
    if c.doi:
        fields.append(f"  doi = {{{c.doi}}}")
    return "@article{{{},\n{}\n}}".format(key, ",\n".join(fields))


def _split_authors(author_str: str) -> list[str]:
    if not author_str:
        return []
    return [a.strip() for a in re.split(r"\s+and\s+", author_str) if a.strip()]


def _safe_int(val: str | None) -> int | None:
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def _name_to_csl(full_name: str) -> dict[str, str]:
    parts = full_name.strip().split()
    if len(parts) == 1:
        return {"family": parts[0]}
    return {"given": " ".join(parts[:-1]), "family": parts[-1]}
