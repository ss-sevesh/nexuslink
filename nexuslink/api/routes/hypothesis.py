"""Hypothesis routes: generation, listing, individual fetch, and reports."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from nexuslink.api.deps import get_nexuslink
from nexuslink.main import NexusLink

router = APIRouter(tags=["hypothesis"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HypothesisRequest(BaseModel):
    top_n: int = Field(default=5, ge=1, le=50)
    skip_validation: bool = False


class HypothesisOut(BaseModel):
    id: str
    statement: str
    confidence: float
    novelty_score: float | None = None
    feasibility_score: float | None = None
    impact_score: float | None = None
    composite_score: float | None = None
    overall_rank: int | None = None
    domains_spanned: list[str] = Field(default_factory=list)
    status: str = "generated"
    evidence_bridges: list[str] = Field(default_factory=list)
    suggested_experiments: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class HypothesisRunResult(BaseModel):
    hypotheses_generated: int
    top_hypothesis: str | None
    top_composite_score: float | None
    domains_covered: int
    report_path: str | None
    hypotheses: list[HypothesisOut]


class ReportSummary(BaseModel):
    id: str
    filename: str
    created_at: str
    size_bytes: int


class ReportDetail(BaseModel):
    id: str
    filename: str
    content: str


# ---------------------------------------------------------------------------
# POST /api/hypothesize
# ---------------------------------------------------------------------------

@router.post("/hypothesize", response_model=HypothesisRunResult)
async def run_hypothesize(
    body: HypothesisRequest,
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
) -> HypothesisRunResult:
    """Trigger the full hypothesis generation, ranking, and report pipeline."""
    from nexuslink.llm.hypothesis.pipeline import run_hypothesis_pipeline  # noqa: PLC0415

    logger.info("POST /api/hypothesize — top_n={}", body.top_n)
    result = await run_hypothesis_pipeline(
        vault_path=nx._vault,
        config=nx._config,
        top_bridges=body.top_n * 4,
        skip_validation=body.skip_validation,
    )

    # After generation, read the freshly written Hypothesis.md files
    hyp_dir = nx._vault / "03-hypotheses"
    file_hyps = await _load_hypotheses_from_vault(hyp_dir)

    return HypothesisRunResult(
        hypotheses_generated=result.get("hypotheses_generated", 0),
        top_hypothesis=result.get("top_hypothesis"),
        top_composite_score=result.get("top_composite_score"),
        domains_covered=result.get("domains_covered", 0),
        report_path=result.get("report_path"),
        hypotheses=file_hyps,
    )


# ---------------------------------------------------------------------------
# GET /api/hypotheses
# ---------------------------------------------------------------------------

@router.get("/hypotheses", response_model=list[HypothesisOut])
async def list_hypotheses(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
    sort_by: str = Query(default="overall_rank", pattern="^(overall_rank|confidence|composite_score)$"),
) -> list[HypothesisOut]:
    """List all hypothesis notes from wiki/03-hypotheses/, sorted by rank."""
    hyp_dir = nx._vault / "03-hypotheses"
    hypotheses = await _load_hypotheses_from_vault(hyp_dir)

    if sort_by == "confidence":
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    elif sort_by == "composite_score":
        hypotheses.sort(key=lambda h: (h.composite_score or 0), reverse=True)
    else:
        hypotheses.sort(key=lambda h: (h.overall_rank or 999))

    return hypotheses


# ---------------------------------------------------------------------------
# GET /api/hypotheses/{id}
# ---------------------------------------------------------------------------

@router.get("/hypotheses/{hypothesis_id}", response_model=HypothesisOut)
async def get_hypothesis(
    hypothesis_id: str,
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
) -> HypothesisOut:
    """Return a single hypothesis with full evidence map and experiment list."""
    hyp_dir = nx._vault / "03-hypotheses"
    # Try exact filename match first, then scan by id field
    candidate = hyp_dir / f"{hypothesis_id}.md"
    if candidate.exists():
        return await _parse_hypothesis_file(candidate)

    # Scan all files for matching id frontmatter
    if hyp_dir.exists():
        for path in hyp_dir.glob("*.md"):
            hyp = await _parse_hypothesis_file(path)
            if hyp.id == hypothesis_id:
                return hyp

    raise HTTPException(status_code=404, detail=f"Hypothesis {hypothesis_id!r} not found.")


# ---------------------------------------------------------------------------
# GET /api/reports
# ---------------------------------------------------------------------------

@router.get("/reports", response_model=list[ReportSummary])
async def list_reports(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
) -> list[ReportSummary]:
    """List all generated report files from wiki/04-reports/."""
    reports_dir = nx._vault / "04-reports"
    if not reports_dir.exists():
        return []

    summaries: list[ReportSummary] = []
    for path in sorted(reports_dir.glob("*.md"), reverse=True):
        stat = path.stat()
        summaries.append(ReportSummary(
            id=path.stem,
            filename=path.name,
            created_at=_mtime_iso(stat.st_mtime),
            size_bytes=stat.st_size,
        ))
    return summaries


# ---------------------------------------------------------------------------
# GET /api/reports/{id}
# ---------------------------------------------------------------------------

@router.get("/reports/{report_id}", response_model=ReportDetail)
async def get_report(
    report_id: str,
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
) -> ReportDetail:
    """Return the full Markdown content of a specific report."""
    reports_dir = nx._vault / "04-reports"
    path = reports_dir / f"{report_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report {report_id!r} not found.")

    content = await asyncio.to_thread(path.read_text, "utf-8")
    return ReportDetail(id=report_id, filename=path.name, content=content)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

async def _load_hypotheses_from_vault(hyp_dir: Path) -> list[HypothesisOut]:
    if not hyp_dir.exists():
        return []
    files = sorted(hyp_dir.glob("*.md"))
    return list(await asyncio.gather(*[_parse_hypothesis_file(f) for f in files]))


async def _parse_hypothesis_file(path: Path) -> HypothesisOut:
    content = await asyncio.to_thread(path.read_text, "utf-8")
    fm, body = _split_frontmatter(content)

    hyp_id: str = str(fm.get("id") or path.stem)
    confidence = float(fm.get("confidence") or 0.5)
    novelty = _optional_float(fm.get("novelty_score"))
    feasibility = _optional_float(fm.get("feasibility_score"))
    impact = _optional_float(fm.get("impact_score"))
    overall_rank = _optional_int(fm.get("overall_rank"))
    domains_spanned = _coerce_list(fm.get("domains_spanned"))
    status: str = str(fm.get("status") or "generated")
    weaknesses = _coerce_list(fm.get("weaknesses"))

    composite: float | None = None
    if novelty is not None and feasibility is not None and impact is not None:
        composite = round(0.4 * novelty + 0.3 * impact + 0.3 * feasibility, 2)

    statement = _extract_section(body, "Hypothesis Statement")
    evidence_bridges = _extract_bridge_list(body, "Evidence From")
    suggested_experiments = _extract_numbered_list(body, "Suggested Experiments")

    return HypothesisOut(
        id=hyp_id,
        statement=statement,
        confidence=confidence,
        novelty_score=novelty,
        feasibility_score=feasibility,
        impact_score=impact,
        composite_score=composite,
        overall_rank=overall_rank,
        domains_spanned=domains_spanned,
        status=status,
        evidence_bridges=evidence_bridges,
        suggested_experiments=suggested_experiments,
        weaknesses=weaknesses,
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


def _extract_section(body: str, heading: str) -> str:
    m = re.search(rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_bridge_list(body: str, heading: str) -> list[str]:
    raw = _extract_section(body, heading)
    return [
        m.group(1).strip()
        for line in raw.splitlines()
        if (m := re.search(r"\[\[([^\]]+)\]\]", line))
    ]


def _extract_numbered_list(body: str, heading: str) -> list[str]:
    raw = _extract_section(body, heading)
    return [
        re.sub(r"^\d+\.\s*", "", line).strip()
        for line in raw.splitlines()
        if re.match(r"^\d+\.", line.strip())
    ]


def _coerce_list(val: object) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str) and val:
        return [val]
    return []


def _optional_float(val: object) -> float | None:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_int(val: object) -> int | None:
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mtime_iso(mtime: float) -> str:
    from datetime import datetime, timezone  # noqa: PLC0415

    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
