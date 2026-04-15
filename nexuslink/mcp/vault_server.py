"""MCP server exposing the NexusLink Obsidian vault to Claude Code.

Tools available:
  search_vault      — full-text search across all notes
  get_note          — read any note by title or filename
  list_concepts     — list concept nodes, optionally filtered by domain
  list_papers       — list ingested papers, optionally filtered by domain
  find_bridges      — find cross-domain concept bridges (from knowledge graph)
  get_connections   — wikilink neighbours of a concept
  get_hypotheses    — list generated hypothesis notes

Run with:
  uv run python -m nexuslink.mcp.vault_server
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Vault paths (relative to this file: mcp/ → nexuslink/ → wiki/)
# ---------------------------------------------------------------------------
_NEXUSLINK_DIR = Path(__file__).parent.parent          # nexuslink/
_WIKI_DIR = _NEXUSLINK_DIR / "wiki"
_PAPERS_DIR = _WIKI_DIR / "01-papers"
_CONCEPTS_DIR = _WIKI_DIR / "02-concepts"
_HYPOTHESES_DIR = _WIKI_DIR / "03-hypotheses"
_CACHE_PICKLE = _WIKI_DIR / ".cache" / "graph.gpickle"

mcp = FastMCP("nexuslink-vault")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_note(path: Path) -> dict[str, Any]:
    """Return {frontmatter, body, title, path} for a markdown note."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    fm: dict = {}
    body = text
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if m:
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            pass
        body = text[m.end():]

    return {
        "frontmatter": fm,
        "body": body,
        "title": fm.get("title") or path.stem,
        "path": str(path.relative_to(_WIKI_DIR)),
    }


def _wikilinks(text: str) -> list[str]:
    return re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]", text)


def _load_graph():
    if not _CACHE_PICKLE.exists():
        return None
    with open(_CACHE_PICKLE, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_vault(query: str, note_type: str = "all") -> str:
    """Full-text search across vault notes.

    Parameters
    ----------
    query:
        Search term(s) — case-insensitive.
    note_type:
        One of "all", "papers", "concepts", "hypotheses".
    """
    dirs: dict[str, Path] = {
        "papers": _PAPERS_DIR,
        "concepts": _CONCEPTS_DIR,
        "hypotheses": _HYPOTHESES_DIR,
    }
    if note_type == "all":
        search_dirs = list(dirs.values())
    elif note_type in dirs:
        search_dirs = [dirs[note_type]]
    else:
        return f"Unknown note_type {note_type!r}. Choose from: all, papers, concepts, hypotheses."

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results: list[str] = []

    for d in search_dirs:
        if not d.exists():
            continue
        for md in sorted(d.glob("*.md")):
            note = _parse_note(md)
            if not note:
                continue
            matches = pattern.findall(note["body"])
            if matches or pattern.search(note["title"]):
                snippet = ""
                for line in note["body"].splitlines():
                    if pattern.search(line):
                        snippet = line.strip()[:120]
                        break
                results.append(f"[{note['path']}] {note['title']}\n  > {snippet}")

    if not results:
        return f"No notes match {query!r}."
    return "\n\n".join(results)


@mcp.tool()
def get_note(title: str) -> str:
    """Retrieve the full content of a vault note by its title or filename stem.

    Searches papers/, concepts/, 03-hypotheses/ in that order.
    """
    candidates = [_PAPERS_DIR, _CONCEPTS_DIR, _HYPOTHESES_DIR]

    # Normalise for fuzzy match
    needle = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip().lower()

    for d in candidates:
        if not d.exists():
            continue
        for md in d.glob("*.md"):
            if md.stem.lower() == needle:
                return md.read_text(encoding="utf-8")

    # Fallback: partial match
    for d in candidates:
        if not d.exists():
            continue
        for md in d.glob("*.md"):
            if needle in md.stem.lower():
                return md.read_text(encoding="utf-8")

    return f"Note {title!r} not found in the vault."


@mcp.tool()
def list_concepts(domain: str = "") -> str:
    """List all concept notes in the vault.

    Parameters
    ----------
    domain:
        Optional filter — only return concepts whose domain list contains this
        value (case-insensitive, partial match).  Leave empty for all concepts.
    """
    if not _CONCEPTS_DIR.exists():
        return "No concepts directory found — run the pipeline first."

    rows: list[str] = []
    for md in sorted(_CONCEPTS_DIR.glob("*.md")):
        note = _parse_note(md)
        if not note:
            continue
        domains: list[str] = note["frontmatter"].get("domains", [])
        if domain and not any(domain.lower() in d.lower() for d in domains):
            continue
        entity_type = note["frontmatter"].get("type", "")
        rows.append(f"- [[{note['title']}]] ({entity_type}) — domains: {', '.join(domains) or 'unknown'}")

    if not rows:
        return f"No concepts found{' for domain ' + domain if domain else ''}."
    return "\n".join(rows)


@mcp.tool()
def list_papers(domain: str = "") -> str:
    """List all ingested paper notes.

    Parameters
    ----------
    domain:
        Optional domain filter (case-insensitive, partial match).
    """
    if not _PAPERS_DIR.exists():
        return "No papers directory found — run the pipeline first."

    rows: list[str] = []
    for md in sorted(_PAPERS_DIR.glob("*.md")):
        note = _parse_note(md)
        if not note:
            continue
        domains: list[str] = note["frontmatter"].get("domain", [])
        if isinstance(domains, str):
            domains = [domains]
        if domain and not any(domain.lower() in d.lower() for d in domains):
            continue
        year = note["frontmatter"].get("year", "")
        rows.append(f"- [[{note['title']}]] ({year}) — domains: {', '.join(domains) or 'unknown'}")

    if not rows:
        return f"No papers found{' for domain ' + domain if domain else ''}."
    return "\n".join(rows)


@mcp.tool()
def find_bridges(domain_a: str = "", domain_b: str = "") -> str:
    """Find cross-domain concept bridges from the knowledge graph.

    Returns bridges sorted by similarity score (strongest first).

    Parameters
    ----------
    domain_a, domain_b:
        Optional domain filters.  If both are given, only bridges that connect
        those two specific domains are returned.  If only one is given, all
        bridges that involve that domain are returned.  If neither is given,
        all bridges are returned.
    """
    graph = _load_graph()
    if graph is None:
        return "Knowledge graph cache not found — run `nexuslink link` first."

    _bridge_relations = {"analogous", "enables", "extends", "contradicts"}
    seen: set[frozenset] = set()
    rows: list[tuple[float, str]] = []

    for src, dst, data in graph.edges(data=True):
        if data.get("relation") not in _bridge_relations:
            continue
        pair = frozenset([src, dst])
        if pair in seen:
            continue
        seen.add(pair)

        da = str(data.get("domain_a", "")).lower()
        db = str(data.get("domain_b", "")).lower()

        if domain_a and domain_b:
            a_lo, b_lo = domain_a.lower(), domain_b.lower()
            if not (
                (a_lo in da and b_lo in db) or (a_lo in db and b_lo in da)
            ):
                continue
        elif domain_a:
            a_lo = domain_a.lower()
            if a_lo not in da and a_lo not in db:
                continue
        elif domain_b:
            b_lo = domain_b.lower()
            if b_lo not in da and b_lo not in db:
                continue

        src_name = graph.nodes[src].get("name", src)
        dst_name = graph.nodes[dst].get("name", dst)
        sim = float(data.get("similarity", 0.0))
        relation = data.get("relation", "related")
        rows.append((sim, f"[[{src_name}]] ({da}) —[{relation} {sim:.2f}]→ [[{dst_name}]] ({db})"))

    if not rows:
        filter_msg = ""
        if domain_a and domain_b:
            filter_msg = f" between {domain_a!r} and {domain_b!r}"
        elif domain_a or domain_b:
            filter_msg = f" for domain {(domain_a or domain_b)!r}"
        return f"No bridges found{filter_msg}."

    rows.sort(reverse=True)
    lines = [f"{i+1}. {row}" for i, (_, row) in enumerate(rows)]
    return f"Found {len(rows)} bridge(s):\n\n" + "\n".join(lines)


@mcp.tool()
def get_connections(concept: str, depth: int = 1) -> str:
    """Return the wikilink neighbours of a concept note.

    Reads the concept's .md file and extracts all [[wikilinks]], then
    optionally follows them one level deeper.

    Parameters
    ----------
    concept:
        Name of the concept (title or filename stem).
    depth:
        1 = direct neighbours only, 2 = neighbours of neighbours.
        Capped at 2 to avoid large outputs.
    """
    depth = min(max(depth, 1), 2)

    def _neighbours(title: str) -> tuple[str, list[str]]:
        content = get_note(title)
        if content.startswith("Note ") and "not found" in content:
            return content, []
        links = _wikilinks(content)
        return content, links

    content, direct = _neighbours(concept)
    if not direct:
        return f"No wikilinks found in note {concept!r} (or note not found)."

    lines = [f"Direct links from [[{concept}]]:", *[f"  - [[{l}]]" for l in direct]]

    if depth == 2:
        for neighbour in direct[:10]:  # cap to avoid huge output
            _, second = _neighbours(neighbour)
            if second:
                lines.append(f"\nLinks from [[{neighbour}]]:")
                lines.extend(f"  - [[{l}]]" for l in second[:8])

    return "\n".join(lines)


@mcp.tool()
def get_hypotheses() -> str:
    """List all generated hypothesis notes in wiki/03-hypotheses/."""
    if not _HYPOTHESES_DIR.exists():
        return "No hypotheses directory found — run the full pipeline first."

    rows: list[str] = []
    for md in sorted(_HYPOTHESES_DIR.glob("*.md")):
        note = _parse_note(md)
        if not note:
            continue
        fm = note["frontmatter"]
        score = fm.get("composite_score", "")
        domains = fm.get("domains_spanned", fm.get("domains", []))
        rows.append(
            f"- [[{note['title']}]] | score: {score} | domains: {', '.join(domains) if isinstance(domains, list) else domains}"
        )

    if not rows:
        return "No hypothesis notes found — run `nexuslink hypothesize` first."
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Smart query tool  (second-brain entry point — token-efficient)
# ---------------------------------------------------------------------------

@mcp.tool()
def ask_vault(question: str) -> str:
    """Answer a research question using only the vault's knowledge graph.

    This is the primary entry point for second-brain queries.  It returns a
    compact, structured digest — not raw note dumps — so Claude can reason
    over it without burning tokens on boilerplate.

    Examples
    --------
    - "find a hypothesis between physics and biology"
    - "what methods appear in both cs and medicine papers?"
    - "which concepts bridge the most domains?"
    - "what do we know about attention mechanisms?"
    """
    q = question.lower()

    # ---- Route: hypothesis / bridge between two domains --------------------
    bridge_match = re.search(
        r"(?:between|connecting|bridge|link)\s+(\w[\w\s]*?)\s+and\s+(\w[\w\s]*?)(?:\?|$|\.)",
        q,
    )
    if bridge_match or any(w in q for w in ("hypothesis", "bridge", "cross-domain", "connect")):
        da = bridge_match.group(1).strip() if bridge_match else ""
        db = bridge_match.group(2).strip() if bridge_match else ""
        bridges_text = find_bridges(da, db)

        # Also list concepts in those domains for context
        context = ""
        if da:
            context += f"\nConcepts in {da}:\n" + list_concepts(da)
        if db:
            context += f"\nConcepts in {db}:\n" + list_concepts(db)

        return f"## Vault answer: cross-domain bridges\n\n{bridges_text}{context}"

    # ---- Route: what do we know about X ------------------------------------
    know_match = re.search(r"(?:about|on|regarding|what is)\s+[\"']?([^?\"']+)[\"']?", q)
    if know_match:
        term = know_match.group(1).strip()
        note_content = get_note(term)
        connections = get_connections(term, depth=1)
        search_hits = search_vault(term)
        # Keep token cost low: truncate raw note to 800 chars
        note_snippet = note_content[:800] + ("…" if len(note_content) > 800 else "")
        return (
            f"## Vault answer: {term}\n\n"
            f"### Note\n{note_snippet}\n\n"
            f"### Wikilink connections\n{connections}\n\n"
            f"### Also appears in\n{search_hits}"
        )

    # ---- Route: methods / overlap across domains --------------------------
    if any(w in q for w in ("method", "technique", "approach", "used in both")):
        domain_match = re.findall(r"\b(physics|biology|chemistry|cs|medicine|materials|engineering|mathematics)\b", q)
        results = []
        for md in sorted(_CONCEPTS_DIR.glob("*.md")):
            note = _parse_note(md)
            if note.get("frontmatter", {}).get("type") != "method":
                continue
            domains = note["frontmatter"].get("domains", [])
            if domain_match and not any(d in " ".join(domains).lower() for d in domain_match):
                continue
            results.append(f"- [[{note['title']}]] — domains: {', '.join(domains)}")
        if not results:
            return "No method concepts found matching that query."
        return "## Vault answer: methods\n\n" + "\n".join(results[:30])

    # ---- Fallback: full-text search + hypotheses listing ------------------
    hits = search_vault(question)
    hyps = get_hypotheses()
    return f"## Vault answer (search)\n\n{hits}\n\n## Existing hypotheses\n\n{hyps}"


# ---------------------------------------------------------------------------
# Write tools  (Claude → Obsidian)
# ---------------------------------------------------------------------------

@mcp.tool()
def add_hypothesis(
    title: str,
    body: str,
    domains: list[str] | None = None,
    composite_score: float = 0.0,
) -> str:
    """Write a new hypothesis note to wiki/03-hypotheses/.

    Parameters
    ----------
    title:
        Title of the hypothesis (used as filename and frontmatter title).
    body:
        Markdown body — can include [[wikilinks]] to concepts and papers.
    domains:
        List of domains this hypothesis spans.
    composite_score:
        0.0–1.0 score for ranking.
    """
    _HYPOTHESES_DIR.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip()[:200]
    path = _HYPOTHESES_DIR / f"{safe_title}.md"

    domains_yaml = "[" + ", ".join(f'"{d}"' for d in (domains or [])) + "]"
    content = f"""\
---
title: "{title.replace('"', '\\"')}"
composite_score: {composite_score:.3f}
domains_spanned: {domains_yaml}
tags: [hypothesis, claude-generated]
---

{body}
"""
    path.write_text(content, encoding="utf-8")
    return f"Hypothesis written to {path.relative_to(_WIKI_DIR)}"


@mcp.tool()
def annotate_concept(concept: str, annotation: str) -> str:
    """Append a Claude annotation block to an existing concept note.

    Parameters
    ----------
    concept:
        Title or filename stem of the concept to annotate.
    annotation:
        Markdown text to append under a '## Claude Notes' section.
    """
    needle = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", concept).strip().lower()

    target: Path | None = None
    for md in _CONCEPTS_DIR.glob("*.md"):
        if md.stem.lower() == needle:
            target = md
            break
    if target is None:
        for md in _CONCEPTS_DIR.glob("*.md"):
            if needle in md.stem.lower():
                target = md
                break

    if target is None:
        return f"Concept {concept!r} not found — cannot annotate."

    existing = target.read_text(encoding="utf-8")
    if "## Claude Notes" in existing:
        # Append inside existing section
        updated = existing.rstrip() + f"\n\n{annotation}\n"
    else:
        updated = existing.rstrip() + f"\n\n## Claude Notes\n\n{annotation}\n"

    target.write_text(updated, encoding="utf-8")
    return f"Annotated {target.relative_to(_WIKI_DIR)}"


@mcp.tool()
def link_concepts(
    concept_a: str,
    concept_b: str,
    relation: str,
    reason: str,
) -> str:
    """Record a Claude-discovered bridge between two concepts in both their notes.

    This adds a wikilink entry under '## Claude Notes' in each concept file
    so the bridge is visible in the Obsidian graph.

    Parameters
    ----------
    concept_a, concept_b:
        Names of the two concepts to link.
    relation:
        Relationship label, e.g. 'analogous', 'enables', 'extends', 'contradicts'.
    reason:
        One-sentence explanation of why these concepts are linked.
    """
    note_a = f"[[{concept_b}]] — {relation}: {reason}"
    note_b = f"[[{concept_a}]] — {relation} (reverse): {reason}"

    result_a = annotate_concept(concept_a, note_a)
    result_b = annotate_concept(concept_b, note_b)
    return f"{result_a}\n{result_b}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
