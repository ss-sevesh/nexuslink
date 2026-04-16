"""Report writer: assembles the final ranked-hypothesis report in Markdown and LaTeX."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nexuslink.llm.hypothesis.generator import _RETRY_DELAYS
from nexuslink.llm.prompts.templates import get_system_prompt, render_template
from nexuslink.llm.scoring.ranker import ScoredHypothesis
from nexuslink.utils.json_parser import extract_json
from nexuslink.wiki.citations.manager import CitationManager
from nexuslink.wiki.graph.builder import KnowledgeGraph

if TYPE_CHECKING:
    from nexuslink.config import NexusConfig

_WIKI_DIR = Path(__file__).parent.parent.parent / "wiki"
_REPORTS_DIR = _WIKI_DIR / "04-reports"
_LATEX_DIR = _REPORTS_DIR / "latex"

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 8192

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]")


class ReportWriter:
    """Generates structured research reports from ranked hypotheses."""

    def __init__(
        self,
        config: "NexusConfig | None" = None,
        *,
        api_key: str | None = None,
    ) -> None:
        import os

        self._ollama_model = (
            (config.ollama_model if config else None)
            or os.environ.get("OLLAMA_MODEL")
        )

        if self._ollama_model:
            logger.info("ReportWriter using local Ollama model: {}", self._ollama_model)
            self._client = None
            self._model = self._ollama_model
        else:
            import anthropic
            effective_key = (config.anthropic_api_key if config else None) or api_key
            if not effective_key:
                logger.warning(
                    "No Anthropic API key set. "
                    "Set ANTHROPIC_API_KEY or OLLAMA_MODEL in .env."
                )
            self._client = anthropic.AsyncAnthropic(api_key=effective_key)
            self._model = _MODEL

    async def generate_report(
        self,
        scored_hypotheses: list[ScoredHypothesis],
        graph: KnowledgeGraph,
        citation_manager: CitationManager,
        vault_path: Path | None = None,
        pipeline_stats: dict | None = None,
    ) -> str:
        """Build the full report, save Markdown + LaTeX, return the Markdown path as str."""
        stats = _build_stats(graph, pipeline_stats or {})

        # Ask Claude for executive summary + cross-domain narrative
        exec_summary, narrative = await self._synthesise(scored_hypotheses, stats)

        md_content = _render_markdown_report(
            scored_hypotheses, exec_summary, narrative, citation_manager, stats
        )

        # Add YAML frontmatter
        md_content = _add_frontmatter(md_content, scored_hypotheses, stats)

        # Validate wikilinks if vault_path supplied
        if vault_path is not None:
            md_content, broken = _validate_wikilinks(md_content, vault_path)
            if broken:
                logger.warning(
                    "Report has {} wikilink(s) with no matching vault note: {}",
                    len(broken), broken[:5],
                )

        latex_content = _render_latex_report(
            scored_hypotheses, exec_summary, narrative, citation_manager, stats
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        reports_dir = (vault_path / "04-reports") if vault_path else _REPORTS_DIR
        latex_dir = reports_dir / "latex"
        md_path = reports_dir / f"report_{timestamp}.md"
        tex_path = latex_dir / f"report_{timestamp}.tex"

        reports_dir.mkdir(parents=True, exist_ok=True)
        latex_dir.mkdir(parents=True, exist_ok=True)

        await asyncio.gather(
            asyncio.to_thread(md_path.write_text, md_content, "utf-8"),
            asyncio.to_thread(tex_path.write_text, latex_content, "utf-8"),
        )

        logger.info("Report saved: {}", md_path)
        logger.info("LaTeX saved:  {}", tex_path)
        return str(md_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int,
    ) -> dict | list:
        if getattr(self, "_ollama_model", None):
            return await self._call_ollama(system, user, max_tokens)

        import anthropic
        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, 1):
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = resp.content[0].text
                try:
                    logger.debug(
                        "Claude usage: input={} output={} tokens",
                        resp.usage.input_tokens,
                        resp.usage.output_tokens,
                    )
                except AttributeError:
                    pass
                return extract_json(text)
            except anthropic.RateLimitError as exc:
                last_exc = exc
                logger.warning("Rate limit (attempt {}), retrying in {}s", attempt, delay)
                await asyncio.sleep(delay)
            except anthropic.APIStatusError as exc:
                last_exc = exc
                logger.error("API error on attempt {}: {} {}", attempt, exc.status_code, exc.message)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
            except Exception as exc:
                last_exc = exc
                logger.error("Synthesis error on attempt {}: {}", attempt, exc)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)

        raise RuntimeError("Claude synthesis failed") from last_exc

    async def _call_ollama(
        self,
        system: str,
        user: str,
        max_tokens: int,
    ) -> dict | list:
        import os
        import httpx
        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
        payload = {
            "model": self._ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": max_tokens}
        }
        
        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, 1):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    text = resp.json()["message"]["content"]
                    return extract_json(text)
            except Exception as exc:
                last_exc = exc
                logger.error("Ollama API error on attempt {}: {}", attempt, exc)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                    
        raise RuntimeError("Ollama API failed after multiple attempts") from last_exc

    async def _synthesise(
        self, hypotheses: list[ScoredHypothesis], stats: dict
    ) -> tuple[str, str]:
        system = get_system_prompt("report_synthesis")
        user = render_template("report_synthesis", hypotheses=hypotheses, stats=stats)

        try:
            data = await self._call_llm(system, user, _MAX_TOKENS)
            if isinstance(data, dict):
                return (
                    data.get("executive_summary", ""),
                    data.get("cross_domain_narrative", ""),
                )
        except Exception as parse_exc:
            logger.warning("Report synthesis parse failed: {}", parse_exc)
            
        return "", ""


# ---------------------------------------------------------------------------
# Markdown report renderer
# ---------------------------------------------------------------------------

def _render_markdown_report(
    hypotheses: list[ScoredHypothesis],
    exec_summary: str,
    narrative: str,
    citations: CitationManager,
    stats: dict,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    exec_summary_str = exec_summary if isinstance(exec_summary, str) else str(exec_summary or "")
    narrative_str = narrative if isinstance(narrative, str) else str(narrative or "")
    lines: list[str] = [
        f"# NexusLink Research Report\n",
        f"*Generated: {timestamp}*\n",
        "---\n",
        "## Executive Summary\n",
        exec_summary_str or "_Executive summary not generated._",
        "\n---\n",
        "## Cross-Domain Analysis Map\n",
        narrative_str or "_Narrative not generated._",
        "\n---\n",
        "## Ranked Hypotheses\n",
    ]

    for h in hypotheses:
        lines.append(f"### Hypothesis {h.overall_rank}: Score {h.composite_score:.1f}/10\n")
        lines.append(f"**Statement:** {h.statement}\n")
        lines.append(
            f"**Domains:** {' × '.join(h.domains_spanned)}"
            f" | N={h.novelty_score:.0f} / F={h.feasibility_score:.0f}"
            f" / I={h.impact_score:.0f} / Overall={h.composite_score:.1f}\n"
        )
        lines.append("**Evidence bridges:**")
        for bridge_key in h.evidence_bridges:
            parts = bridge_key.split("::")
            if len(parts) == 2:
                lines.append(f"- [[{parts[0]}]] ↔ [[{parts[1]}]]")
            else:
                lines.append(f"- {bridge_key}")
        lines.append("\n**Suggested experiments:**")
        for i, exp in enumerate(h.suggested_experiments, 1):
            lines.append(f"{i}. {exp if isinstance(exp, str) else str(exp)}")
        if h.weaknesses:
            lines.append("\n**Known weaknesses:**")
            for w in h.weaknesses:
                lines.append(f"- {w if isinstance(w, str) else str(w)}")
        lines.append("")

    lines += [
        "---\n",
        "## Evidence Graph Description\n",
        f"The knowledge graph contains **{stats['total_concepts']} concepts** "
        f"spanning **{stats['domains_covered']} domains** "
        f"with **{stats['total_bridges']} cross-domain bridges** detected.\n",
        f"Domains covered: {', '.join(f'[[{d}]]' for d in stats.get('domains', []))}\n",
        "---\n",
        "## Full Bibliography\n",
        "```bibtex",
        citations.to_bibtex() or "% No citations recorded yet.",
        "```\n",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX report renderer
# ---------------------------------------------------------------------------

def _render_latex_report(
    hypotheses: list[ScoredHypothesis],
    exec_summary: str,
    narrative: str,
    citations: CitationManager,
    stats: dict,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    hyp_sections: list[str] = []
    for h in hypotheses:
        domains_str = " $\\times$ ".join(_tex(d) for d in h.domains_spanned)
        experiments = "\n".join(
            f"  \\item {_tex(exp)}" for exp in h.suggested_experiments
        )
        weaknesses = (
            "\n".join(f"  \\item {_tex(w)}" for w in h.weaknesses)
            if h.weaknesses
            else "  \\item None identified."
        )
        hyp_sections.append(
            f"""\
\\subsection*{{Hypothesis {h.overall_rank} (Score: {h.composite_score:.1f}/10)}}

\\textbf{{Statement:}} {_tex(h.statement)}

\\noindent\\textbf{{Domains:}} {domains_str}
\\hspace{{1em}} N={h.novelty_score:.0f} / F={h.feasibility_score:.0f} / I={h.impact_score:.0f}

\\textbf{{Suggested experiments:}}
\\begin{{enumerate}}
{experiments}
\\end{{enumerate}}

\\textbf{{Known weaknesses:}}
\\begin{{itemize}}
{weaknesses}
\\end{{itemize}}
"""
        )

    csl_refs: list[str] = []
    for item in citations.to_csl_json():
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in item.get("author", [])
        )
        year = item.get("issued", {}).get("date-parts", [[""]])[0][0]
        title = item.get("title", "")
        doi = item.get("DOI", "")
        csl_refs.append(
            f"\\bibitem{{{_tex(item['id'])}}}\n"
            f"{_tex(authors)} ({year}). \\textit{{{_tex(title)}}}."
            + (f" DOI: {_tex(doi)}." if doi else "")
        )

    bib_section = (
        "\n\n".join(csl_refs) if csl_refs else "\\textit{No citations recorded yet.}"
    )

    return rf"""\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{geometry}}
\geometry{{margin=2.5cm}}
\usepackage{{booktabs}}
\usepackage{{microtype}}

\title{{NexusLink Cross-Domain Research Report}}
\author{{Generated by NexusLink}}
\date{{{timestamp}}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

\section{{Executive Summary}}

{_tex(exec_summary) or r"\textit{Not generated.}"}

\section{{Cross-Domain Analysis}}

{_tex(narrative) or r"\textit{Not generated.}"}

\section{{Knowledge Graph Statistics}}

\begin{{tabular}}{{ll}}
\toprule
Metric & Value \\
\midrule
Papers analysed   & {stats.get('papers_processed', 0)} \\
Concepts extracted & {stats['total_concepts']} \\
Cross-domain bridges & {stats['total_bridges']} \\
Domains covered   & {stats['domains_covered']} \\
\bottomrule
\end{{tabular}}

\section{{Ranked Hypotheses}}

{"".join(hyp_sections)}

\section{{References}}

\begin{{thebibliography}}{{99}}
{bib_section}
\end{{thebibliography}}

\end{{document}}
"""


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _add_frontmatter(
    content: str,
    hypotheses: list[ScoredHypothesis],
    stats: dict,
) -> str:
    """Prepend YAML frontmatter to *content*."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    domains = sorted({d for h in hypotheses for d in h.domains_spanned})
    top_score = round(hypotheses[0].composite_score, 2) if hypotheses else 0.0

    fm = (
        f"---\n"
        f'title: "NexusLink Research Report"\n'
        f'date: "{date}"\n'
        f"hypothesis_count: {len(hypotheses)}\n"
        f"domains_covered: [{', '.join(f'\"{d}\"' for d in domains)}]\n"
        f"top_score: {top_score}\n"
        f"tags: []\n"
        f"---\n\n"
    )
    return fm + content


def _validate_wikilinks(content: str, vault_path: Path) -> tuple[str, list[str]]:
    """Strip [[wikilinks]] that have no matching vault note.

    Returns *(cleaned_content, broken_targets)* — broken wikilinks are
    replaced with plain text so they don't appear as ghost nodes in Obsidian.
    """
    _SEARCH_DIRS = ("01-papers", "02-concepts", "03-hypotheses", "04-reports", "")

    broken: list[str] = []

    def _replace(m: re.Match) -> str:
        target = m.group(1).strip()
        found = any(
            (vault_path / subdir / f"{target}.md").exists()
            for subdir in _SEARCH_DIRS
        )
        if found:
            return m.group(0)  # keep as-is
        broken.append(target)
        return target  # strip [[ ]] — leave plain text

    cleaned = _WIKILINK_RE.sub(_replace, content)
    return cleaned, broken


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _build_stats(graph: KnowledgeGraph, pipeline_stats: dict) -> dict:
    node_domains: set[str] = set()
    for _, attrs in graph._graph.nodes(data=True):
        if attrs.get("type") == "concept":
            node_domains.update(attrs.get("domains", []))

    return {
        "total_concepts": graph.node_count("concept"),
        "total_bridges": len(graph.get_bridges()),
        "domains_covered": len(node_domains),
        "domains": sorted(node_domains),
        "papers_processed": pipeline_stats.get("papers_processed", 0),
    }


def _tex(s: str) -> str:
    """Escape LaTeX special characters and strip Obsidian [[wikilinks]].

    Order: strip wikilinks first, then escape specials, then restore backslash.
    """
    import re as _re
    # Strip [[wikilinks]] — keep the display text (or target if no pipe)
    s = _re.sub(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', lambda m: m.group(2) or m.group(1), s)

    _PLACEHOLDER = "\x00BSLASH\x00"
    s = s.replace("\\", _PLACEHOLDER)
    for char, replacement in [
        ("{", r"\{"), ("}", r"\}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        s = s.replace(char, replacement)
    return s.replace(_PLACEHOLDER, r"\textbackslash{}")
