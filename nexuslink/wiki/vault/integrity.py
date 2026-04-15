"""Evidence Integrity Verification — checks retraction status, citation accuracy, evidence reliability.
This is NexusLink's key differentiator: no other hypothesis generation system verifies evidence chains."""

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None


@dataclass
class RetractionStatus:
    doi: str
    is_retracted: bool = False
    retraction_date: str = ""
    retraction_reason: str = ""
    retraction_source: str = ""
    checked_at: str = ""


@dataclass
class CitationIssue:
    paper_title: str
    issue_type: str  # "retracted" | "not_found" | "doi_invalid" | "self_referential"
    severity: str  # "critical" | "warning" | "info"
    details: str = ""


@dataclass
class EvidenceScore:
    paper_doi: str
    paper_title: str
    retraction_clean: bool = True  # not retracted
    citation_count: int = 0
    influential_citation_count: int = 0
    year: int = 0
    has_open_access: bool = False
    reliability_score: float = 1.0  # 0.0 to 1.0


@dataclass
class HypothesisIntegrity:
    hypothesis_id: str
    evidence_scores: list[EvidenceScore] = field(default_factory=list)
    retraction_flags: list[RetractionStatus] = field(default_factory=list)
    citation_issues: list[CitationIssue] = field(default_factory=list)
    overall_integrity_score: float = 1.0  # 0.0 to 1.0
    checked_at: str = ""

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "overall_integrity_score": self.overall_integrity_score,
            "evidence_count": len(self.evidence_scores),
            "retraction_flags": len([r for r in self.retraction_flags if r.is_retracted]),
            "citation_issues": len(self.citation_issues),
            "checked_at": self.checked_at,
        }


class EvidenceIntegrityChecker:
    """Checks the reliability of evidence chains supporting hypotheses."""

    CROSSREF_BASE = "https://api.crossref.org/works"
    S2_BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, vault_path: Path, s2_api_key: str | None = None):
        self.vault_path = Path(vault_path)
        self.cache_dir = self.vault_path / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retraction_cache_path = self.cache_dir / "retraction_cache.json"
        self.integrity_cache_path = self.cache_dir / "integrity_scores.json"
        self.s2_api_key = s2_api_key
        self._retraction_cache = self._load_retraction_cache()
        self._client = None

    def _get_client(self) -> "httpx.Client":
        if not httpx:
            raise ImportError("httpx required: uv add httpx")
        if self._client is None:
            headers = {"User-Agent": "NexusLink/1.0 (mailto:nexuslink@research.org)"}
            if self.s2_api_key:
                headers["x-api-key"] = self.s2_api_key
            self._client = httpx.Client(headers=headers, timeout=30.0)
        return self._client

    def _load_retraction_cache(self) -> dict:
        if self.retraction_cache_path.exists():
            try:
                return json.loads(self.retraction_cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_retraction_cache(self):
        self.retraction_cache_path.write_text(json.dumps(self._retraction_cache, indent=2), encoding="utf-8")

    # --- Retraction Checking via CrossRef ---

    def check_retraction_crossref(self, doi: str) -> RetractionStatus:
        """Check if a DOI has been retracted via CrossRef API (includes Retraction Watch data)."""
        status = RetractionStatus(doi=doi, checked_at=datetime.now().isoformat())

        if doi in self._retraction_cache:
            cached = self._retraction_cache[doi]
            status.is_retracted = cached.get("is_retracted", False)
            status.retraction_reason = cached.get("retraction_reason", "")
            status.retraction_source = "cache"
            return status

        if not doi or not doi.strip():
            return status

        try:
            client = self._get_client()
            resp = client.get(f"{self.CROSSREF_BASE}/{doi}")
            if resp.status_code == 200:
                data = resp.json().get("message", {})
                # CrossRef includes update-to field for retractions
                updates = data.get("update-to", [])
                for update in updates:
                    if update.get("type") == "retraction":
                        status.is_retracted = True
                        status.retraction_reason = update.get("label", "retracted")
                        status.retraction_date = update.get("updated", {}).get("date-time", "")
                        status.retraction_source = "crossref"
                        break

                # Also check if this paper itself is a retraction notice
                doc_type = data.get("type", "")
                if doc_type == "retraction":
                    status.is_retracted = True
                    status.retraction_source = "crossref-type"

            elif resp.status_code == 404:
                logger.debug(f"DOI not found in CrossRef: {doi}")
            else:
                logger.warning(f"CrossRef returned {resp.status_code} for {doi}")

        except Exception as e:
            logger.warning(f"CrossRef check failed for {doi}: {e}")

        self._retraction_cache[doi] = {
            "is_retracted": status.is_retracted,
            "retraction_reason": status.retraction_reason,
            "checked_at": status.checked_at,
        }
        self._save_retraction_cache()
        return status

    # --- Evidence Scoring via Semantic Scholar ---

    def score_paper_evidence(self, doi: str = "", title: str = "") -> EvidenceScore:
        """Score a paper's reliability using S2 metadata."""
        score = EvidenceScore(paper_doi=doi, paper_title=title)

        identifier = f"DOI:{doi}" if doi else None
        if not identifier and title:
            # Search by title
            try:
                client = self._get_client()
                resp = client.get(
                    f"{self.S2_BASE}/paper/search",
                    params={"query": title, "limit": 1, "fields": "title,externalIds,citationCount,influentialCitationCount,year,isOpenAccess"},
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data:
                        paper = data[0]
                        score.citation_count = paper.get("citationCount", 0) or 0
                        score.influential_citation_count = paper.get("influentialCitationCount", 0) or 0
                        score.year = paper.get("year", 0) or 0
                        score.has_open_access = paper.get("isOpenAccess", False) or False
                        ext_ids = paper.get("externalIds", {}) or {}
                        if not doi and ext_ids.get("DOI"):
                            score.paper_doi = ext_ids["DOI"]
                            doi = ext_ids["DOI"]
            except Exception as e:
                logger.debug(f"S2 title search failed for '{title}': {e}")
        elif identifier:
            try:
                client = self._get_client()
                resp = client.get(
                    f"{self.S2_BASE}/paper/{identifier}",
                    params={"fields": "title,citationCount,influentialCitationCount,year,isOpenAccess"},
                )
                if resp.status_code == 200:
                    paper = resp.json()
                    score.citation_count = paper.get("citationCount", 0) or 0
                    score.influential_citation_count = paper.get("influentialCitationCount", 0) or 0
                    score.year = paper.get("year", 0) or 0
                    score.has_open_access = paper.get("isOpenAccess", False) or False
            except Exception as e:
                logger.debug(f"S2 lookup failed for {identifier}: {e}")

        # Check retraction
        if doi:
            retraction = self.check_retraction_crossref(doi)
            score.retraction_clean = not retraction.is_retracted

        # Compute reliability score (0.0 - 1.0)
        score.reliability_score = self._compute_reliability(score)
        return score

    def _compute_reliability(self, score: EvidenceScore) -> float:
        """Reliability = weighted combination of retraction status, citations, recency, openness."""
        if not score.retraction_clean:
            return 0.0  # retracted = zero reliability

        reliability = 0.5  # base

        # Citation boost (log scale, max +0.25)
        import math
        if score.citation_count > 0:
            citation_factor = min(math.log10(score.citation_count + 1) / 4.0, 0.25)
            reliability += citation_factor

        # Influential citation boost (max +0.1)
        if score.influential_citation_count > 0:
            reliability += min(score.influential_citation_count / 50.0, 0.1)

        # Recency boost: papers from last 5 years get +0.1
        current_year = datetime.now().year
        if score.year and (current_year - score.year) <= 5:
            reliability += 0.1

        # Open access small boost (+0.05)
        if score.has_open_access:
            reliability += 0.05

        return min(reliability, 1.0)

    # --- Hypothesis-Level Integrity ---

    def check_hypothesis_integrity(self, hypothesis_path: Path) -> HypothesisIntegrity:
        """Full integrity check for one hypothesis."""
        content = hypothesis_path.read_text(encoding="utf-8")

        # Extract hypothesis ID from YAML
        id_match = re.search(r'^id:\s*"?(\w+)"?', content, re.MULTILINE)
        h_id = id_match.group(1) if id_match else hypothesis_path.stem

        integrity = HypothesisIntegrity(
            hypothesis_id=h_id,
            checked_at=datetime.now().isoformat(),
        )

        # Find all linked papers
        paper_links = re.findall(r'\[\[(.+?)\]\]', content)

        # Read paper notes to get DOIs
        papers_dir = self.vault_path / "01-papers"
        for link_name in paper_links:
            paper_path = papers_dir / f"{link_name}.md"
            if not paper_path.exists():
                continue

            paper_content = paper_path.read_text(encoding="utf-8")
            doi_match = re.search(r'^doi:\s*"?([^\s"]+)"?', paper_content, re.MULTILINE)
            doi = doi_match.group(1) if doi_match else ""
            title_match = re.search(r'^title:\s*"?(.+?)"?\s*$', paper_content, re.MULTILINE)
            title = title_match.group(1) if title_match else link_name

            # Score this paper
            evidence = self.score_paper_evidence(doi=doi, title=title)
            integrity.evidence_scores.append(evidence)

            # Check retraction
            if doi:
                retraction = self.check_retraction_crossref(doi)
                if retraction.is_retracted:
                    integrity.retraction_flags.append(retraction)
                    integrity.citation_issues.append(CitationIssue(
                        paper_title=title,
                        issue_type="retracted",
                        severity="critical",
                        details=f"Paper retracted: {retraction.retraction_reason}",
                    ))

        # Compute overall integrity score
        if integrity.evidence_scores:
            scores = [e.reliability_score for e in integrity.evidence_scores]
            # If ANY evidence is retracted, heavily penalize
            retracted_count = len([r for r in integrity.retraction_flags if r.is_retracted])
            avg_score = sum(scores) / len(scores)
            penalty = retracted_count * 0.3
            integrity.overall_integrity_score = max(0.0, avg_score - penalty)
        else:
            integrity.overall_integrity_score = 0.5  # no evidence = uncertain

        return integrity

    def check_all_hypotheses(self) -> list[HypothesisIntegrity]:
        """Check integrity of all hypotheses in the vault."""
        results = []
        hyp_dir = self.vault_path / "03-hypotheses"
        if not hyp_dir.exists():
            return results

        for hyp_path in sorted(hyp_dir.glob("*.md")):
            logger.info(f"Checking integrity: {hyp_path.name}")
            integrity = self.check_hypothesis_integrity(hyp_path)
            results.append(integrity)

            # Update hypothesis note with integrity score
            self._update_hypothesis_integrity(hyp_path, integrity)

        # Save all results
        self._save_integrity_results(results)
        return results

    def _update_hypothesis_integrity(self, hyp_path: Path, integrity: HypothesisIntegrity):
        """Add/update evidence integrity section in hypothesis note."""
        content = hyp_path.read_text(encoding="utf-8")

        integrity_section = f"""
## Evidence Integrity
- **Overall Integrity Score**: {integrity.overall_integrity_score:.2f}
- **Evidence Sources Checked**: {len(integrity.evidence_scores)}
- **Retraction Flags**: {len([r for r in integrity.retraction_flags if r.is_retracted])}
- **Citation Issues**: {len(integrity.citation_issues)}
- **Last Checked**: {integrity.checked_at}
"""
        if integrity.retraction_flags:
            integrity_section += "\n### Retracted Sources (CRITICAL)\n"
            for r in integrity.retraction_flags:
                if r.is_retracted:
                    integrity_section += f"- ⚠️ DOI:{r.doi} — {r.retraction_reason}\n"

        if integrity.citation_issues:
            integrity_section += "\n### Citation Issues\n"
            for issue in integrity.citation_issues:
                integrity_section += f"- [{issue.severity.upper()}] {issue.paper_title}: {issue.details}\n"

        # Replace or append
        if "## Evidence Integrity" in content:
            content = re.sub(
                r'## Evidence Integrity\n.*?(?=\n## |\Z)',
                integrity_section.strip(),
                content, flags=re.DOTALL,
            )
        else:
            # Insert before ## Related Hypotheses or at end
            if "## Related Hypotheses" in content:
                content = content.replace("## Related Hypotheses", f"{integrity_section}\n## Related Hypotheses")
            else:
                content += f"\n{integrity_section}"

        hyp_path.write_text(content, encoding="utf-8")

    def _save_integrity_results(self, results: list[HypothesisIntegrity]):
        data = [r.to_dict() for r in results]
        self.integrity_cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Integrity results saved: {len(results)} hypotheses checked")

    def get_integrity_summary(self) -> dict:
        """Return summary of vault-wide evidence integrity."""
        if self.integrity_cache_path.exists():
            data = json.loads(self.integrity_cache_path.read_text(encoding="utf-8"))
            total = len(data)
            avg_score = sum(d["overall_integrity_score"] for d in data) / total if total else 0
            flagged = sum(1 for d in data if d["retraction_flags"] > 0)
            return {
                "total_hypotheses_checked": total,
                "average_integrity_score": round(avg_score, 3),
                "hypotheses_with_retraction_flags": flagged,
                "clean_hypotheses": total - flagged,
            }
        return {"total_hypotheses_checked": 0}
