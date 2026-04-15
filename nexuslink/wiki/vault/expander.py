"""Autonomous vault expansion — searches for supporting/refuting papers."""

import re
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None

from nexuslink.wiki.vault.reader import VaultReader
from nexuslink.wiki.vault.models import HypothesisNote


class ExpansionReport:
    def __init__(self):
        self.papers_found: int = 0
        self.papers_ingested: int = 0
        self.supporting: list[str] = []
        self.refuting: list[str] = []


class AutonomousExpander:
    S2_BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, reader: VaultReader, api_key: str | None = None):
        self.reader = reader
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if not httpx:
            raise ImportError("httpx required: uv add httpx")
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._client = httpx.Client(headers=headers, timeout=30.0)
        return self._client

    def _extract_concepts(self, hypothesis: HypothesisNote) -> list[str]:
        concepts = re.findall(r'\[\[(.+?)\]\]', hypothesis.raw_content)
        if not concepts and hypothesis.statement:
            words = hypothesis.statement.split()
            concepts = [" ".join(words[i:i+3]) for i in range(0, min(len(words), 12), 3)]
        return concepts[:5]

    def search_s2(self, query: str, limit: int = 5, fields: str = "title,abstract,externalIds,fieldsOfStudy,citationCount") -> list[dict]:
        try:
            client = self._get_client()
            resp = client.get(
                f"{self.S2_BASE}/paper/search",
                params={"query": query, "limit": limit, "fields": fields},
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"S2 search failed for '{query}': {e}")
            return []

    def find_supporting_papers(self, hypothesis: HypothesisNote, max_papers: int = 5) -> list[dict]:
        concepts = self._extract_concepts(hypothesis)
        if not concepts:
            return []
        query = " ".join(concepts[:3])
        results = self.search_s2(query, limit=max_papers)
        existing_domains = set(hypothesis.domains_spanned)
        filtered = []
        for paper in results:
            paper_fields = set(paper.get("fieldsOfStudy") or [])
            if paper_fields and not paper_fields.issubset(existing_domains):
                filtered.append(paper)
        return filtered[:max_papers]

    def find_refuting_papers(self, hypothesis: HypothesisNote, max_papers: int = 3) -> list[dict]:
        concepts = self._extract_concepts(hypothesis)
        if not concepts:
            return []
        negation_terms = ["failure of", "limitation of", "does not"]
        query = f"{negation_terms[0]} {concepts[0]}" if concepts else ""
        return self.search_s2(query, limit=max_papers)

    def expand_vault_for_hypothesis(self, hypothesis: HypothesisNote) -> ExpansionReport:
        report = ExpansionReport()
        supporting = self.find_supporting_papers(hypothesis)
        refuting = self.find_refuting_papers(hypothesis)

        report.papers_found = len(supporting) + len(refuting)
        report.supporting = [p.get("title", "untitled") for p in supporting]
        report.refuting = [p.get("title", "untitled") for p in refuting]

        # Update hypothesis note with findings
        if (supporting or refuting) and hypothesis.path.exists():
            content = hypothesis.path.read_text()

            if supporting:
                support_section = "\n## Supporting Evidence Found\n"
                for p in supporting:
                    title = p.get("title", "untitled")
                    support_section += f"- {title}\n"
                if "## Supporting Evidence Found" in content:
                    content = re.sub(
                        r'## Supporting Evidence Found\n.*?(?=\n## |\Z)',
                        support_section.strip(),
                        content, flags=re.DOTALL,
                    )
                else:
                    content += f"\n{support_section}"

            if refuting:
                refute_section = "\n## Potential Contradictions\n"
                for p in refuting:
                    title = p.get("title", "untitled")
                    refute_section += f"- {title}\n"
                if "## Potential Contradictions" in content:
                    content = re.sub(
                        r'## Potential Contradictions\n.*?(?=\n## |\Z)',
                        refute_section.strip(),
                        content, flags=re.DOTALL,
                    )
                else:
                    content += f"\n{refute_section}"

            hypothesis.path.write_text(content)

        report.papers_ingested = len(supporting) + len(refuting)
        logger.info(f"Expanded {hypothesis.id}: {len(supporting)} supporting, {len(refuting)} refuting")
        return report

    def suggest_next_domains(self) -> list[str]:
        try:
            from nexuslink.wiki.taxonomy.classifier import _DOMAIN_KEYWORDS
        except ImportError:
            return []
        papers = self.reader.read_all_papers()
        present_domains = {p.domain for p in papers if p.domain}
        all_domains = set(_DOMAIN_KEYWORDS.keys())
        missing = all_domains - present_domains
        return sorted(missing)

    def auto_expand_cycle(self, max_new_papers: int = 10) -> dict:
        hypotheses = self.reader.read_all_hypotheses()
        targets = [h for h in hypotheses if h.status in ("reviewed", "validated", "generated")]
        targets.sort(key=lambda h: h.composite_score, reverse=True)

        total_expanded = 0
        total_found = 0
        for h in targets[:5]:
            if total_expanded >= max_new_papers:
                break
            report = self.expand_vault_for_hypothesis(h)
            total_found += report.papers_found
            total_expanded += report.papers_ingested

        suggestions = self.suggest_next_domains()
        return {
            "hypotheses_expanded": min(len(targets), 5),
            "papers_found": total_found,
            "papers_ingested": total_expanded,
            "suggested_domains": suggestions,
        }
