"""API route tests — uses FastAPI TestClient with mocked NexusLink dependency."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nexuslink.api.app import app
from nexuslink.api.deps import get_nexuslink

# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------

def _make_nx(tmp_path: Path) -> MagicMock:
    """Build a NexusLink mock pre-wired to a temp vault."""
    vault = tmp_path / "wiki"
    (vault / "papers").mkdir(parents=True)
    (vault / "concepts").mkdir(parents=True)
    (vault / "03-hypotheses").mkdir(parents=True)
    (vault / "04-reports").mkdir(parents=True)

    nx = MagicMock()
    nx._vault = vault
    nx._config = MagicMock()
    nx._config.similarity_threshold = 0.65
    nx._config.anthropic_api_key = "test-key"
    nx._config.top_n_hypotheses = 5
    return nx


# ---------------------------------------------------------------------------
# Fixture: client with mocked NexusLink
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    mock_nx = _make_nx(tmp_path)

    # Ingest responses
    mock_nx.ingest = AsyncMock(return_value={
        "id": "abc123",
        "title": "Test Paper",
        "source_type": "arxiv",
        "entities_found": 3,
        "domain_tags": ["physics"],
        "wiki_note": "papers/Test Paper.md",
    })

    # Status response
    mock_nx.status = AsyncMock(return_value={
        "total_papers": 2,
        "total_concepts": 8,
        "total_bridges": 3,
        "total_hypotheses": 5,
        "domains_covered": ["biology", "physics"],
        "total_vault_notes": 15,
    })

    app.dependency_overrides[get_nexuslink] = lambda: mock_nx

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Ingest routes
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_arxiv_id(self, client: TestClient) -> None:
        resp = client.post("/api/ingest", data={"arxiv_id": "2101.03961"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == "abc123"
        assert body["title"] == "Test Paper"
        assert body["entities_found"] == 3

    def test_ingest_doi(self, client: TestClient) -> None:
        resp = client.post("/api/ingest", data={"doi": "10.1038/nature12345"})
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == "abc123"

    def test_ingest_pdf_upload(self, client: TestClient, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with pdf.open("rb") as f:
            resp = client.post("/api/ingest", files={"file": ("paper.pdf", f, "application/pdf")})
        assert resp.status_code == 200
        assert resp.json()["title"] == "Test Paper"

    def test_ingest_missing_source_returns_400(self, client: TestClient) -> None:
        resp = client.post("/api/ingest")
        assert resp.status_code == 400

    def test_ingest_batch(self, client: TestClient) -> None:
        resp = client.post(
            "/api/ingest/batch",
            json={"sources": ["2101.03961", "10.1038/nature12345"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_ingested"] == 2
        assert len(body["results"]) == 2
        assert body["failed"] == []

    def test_list_papers_empty_vault(self, client: TestClient) -> None:
        resp = client.get("/api/papers")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["papers"] == []

    def test_list_papers_with_notes(self, client: TestClient, tmp_path: Path) -> None:
        vault = tmp_path / "wiki"
        paper = vault / "papers" / "My Paper.md"
        paper.write_text(
            "---\ntitle: \"My Paper\"\nauthors: [\"Alice\"]\ndomain: [\"physics\"]\nyear: 2024\ntags: []\n---\n\n"
            "## Summary\n\n## Entities\n\n- [[CRISPR]] (method)\n",
            "utf-8",
        )
        resp = client.get("/api/papers")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["papers"][0]["title"] == "My Paper"
        assert body["papers"][0]["entity_count"] == 1
        assert body["papers"][0]["domain"] == ["physics"]

    def test_list_papers_pagination(self, client: TestClient, tmp_path: Path) -> None:
        vault = tmp_path / "wiki"
        for i in range(5):
            p = vault / "papers" / f"Paper{i}.md"
            p.write_text(
                f"---\ntitle: \"Paper{i}\"\nauthors: []\ndomain: []\nyear:\ntags: []\n---\n\n"
                "## Entities\n\n<!-- none detected -->\n",
                "utf-8",
            )
        resp = client.get("/api/papers?page=1&page_size=2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 5
        assert len(body["papers"]) == 2

        resp2 = client.get("/api/papers?page=3&page_size=2")
        assert len(resp2.json()["papers"]) == 1


# ---------------------------------------------------------------------------
# Graph routes
# ---------------------------------------------------------------------------

class TestGraph:
    def test_graph_stats(self, client: TestClient) -> None:
        resp = client.get("/api/graph/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_papers"] == 2
        assert body["total_bridges"] == 3
        assert "physics" in body["domains_covered"]

    def test_bridges_no_graph_returns_404(self, client: TestClient) -> None:
        # No graph.gpickle exists in mock vault
        resp = client.get("/api/bridges")
        assert resp.status_code == 404

    def test_bridges_with_graph(self, client: TestClient, tmp_path: Path) -> None:
        from nexuslink.wiki.linker.bridge_finder import ConceptBridge  # noqa: PLC0415
        from nexuslink.wiki.graph.builder import KnowledgeGraph  # noqa: PLC0415

        vault = tmp_path / "wiki"
        cache = vault / ".cache"
        cache.mkdir(parents=True)

        kg = KnowledgeGraph()
        bridge = ConceptBridge(
            entity_a="Casimir effect",
            entity_b="van der Waals forces",
            similarity_score=0.87,
            domain_a="physics",
            domain_b="biology",
            bridge_type="analogous",
        )
        kg.add_bridge(bridge)
        kg.save(cache / "graph.gpickle")

        resp = client.get("/api/bridges")
        assert resp.status_code == 200
        bridges = resp.json()
        assert len(bridges) == 1
        assert bridges[0]["entity_a"] == "Casimir effect"
        assert bridges[0]["similarity_score"] == pytest.approx(0.87)

    def test_bridges_domain_filter(self, client: TestClient, tmp_path: Path) -> None:
        from nexuslink.wiki.linker.bridge_finder import ConceptBridge  # noqa: PLC0415
        from nexuslink.wiki.graph.builder import KnowledgeGraph  # noqa: PLC0415

        vault = tmp_path / "wiki"
        cache = vault / ".cache"
        cache.mkdir(parents=True)

        kg = KnowledgeGraph()
        for ea, eb, da, db in [
            ("A", "B", "physics", "biology"),
            ("C", "D", "chemistry", "medicine"),
        ]:
            kg.add_bridge(ConceptBridge(
                entity_a=ea, entity_b=eb,
                similarity_score=0.75,
                domain_a=da, domain_b=db,
                bridge_type="enables",
            ))
        kg.save(cache / "graph.gpickle")

        resp = client.get("/api/bridges?domain_a=physics")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["entity_a"] == "A"

    def test_graph_export_no_graph_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/graph/export")
        assert resp.status_code == 404

    def test_post_link_triggers_pipeline(self, client: TestClient) -> None:
        with patch(
            "nexuslink.wiki.linker.pipeline.run_linking",
            new=AsyncMock(return_value={
                "total_bridges": 2,
                "total_concepts": 6,
                "domains_covered": 3,
                "papers_processed": 4,
                "concept_notes_written": 6,
            }),
        ):
            resp = client.post("/api/link")
        assert resp.status_code == 200
        body = resp.json()
        assert body["bridges_found"] == 2
        assert body["papers_processed"] == 4


# ---------------------------------------------------------------------------
# Hypothesis routes
# ---------------------------------------------------------------------------

class TestHypothesis:
    def test_list_hypotheses_empty(self, client: TestClient) -> None:
        resp = client.get("/api/hypotheses")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_hypotheses_reads_vault(self, client: TestClient, tmp_path: Path) -> None:
        vault = tmp_path / "wiki"
        hyp_dir = vault / "03-hypotheses"
        hyp_dir.mkdir(parents=True, exist_ok=True)
        (hyp_dir / "H001.md").write_text(
            "---\n"
            'id: "H001"\n'
            "confidence: 0.82\n"
            "novelty_score: 7.5\n"
            "feasibility_score: 6.0\n"
            "impact_score: 8.0\n"
            "overall_rank: 1\n"
            "domains_spanned: [\"physics\", \"biology\"]\n"
            "status: generated\n"
            "weaknesses: []\n"
            "tags: []\n"
            "---\n\n"
            "## Hypothesis Statement\n\n"
            "Nano-surfaces will achieve gecko-level adhesion via Casimir forces.\n\n"
            "## Evidence From\n\n"
            "- [[Casimir effect ↔ van der Waals forces]]\n\n"
            "## Suggested Experiments\n\n"
            "1. Fabricate nano-patterned gold surfaces.\n"
            "2. Measure adhesion with AFM.\n\n"
            "## References\n\n",
            "utf-8",
        )

        resp = client.get("/api/hypotheses")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        hyp = items[0]
        assert hyp["id"] == "H001"
        assert hyp["confidence"] == pytest.approx(0.82)
        assert hyp["overall_rank"] == 1
        assert "Nano-surfaces" in hyp["statement"]
        assert "physics" in hyp["domains_spanned"]
        assert len(hyp["suggested_experiments"]) == 2

    def test_get_hypothesis_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/hypotheses/HXXX")
        assert resp.status_code == 404

    def test_list_reports_empty(self, client: TestClient) -> None:
        resp = client.get("/api/reports")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_report_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/reports/nonexistent")
        assert resp.status_code == 404

    def test_get_report_content(self, client: TestClient, tmp_path: Path) -> None:
        vault = tmp_path / "wiki"
        reports_dir = vault / "04-reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        md = reports_dir / "report_20240101_120000.md"
        md.write_text(
            "# NexusLink Research Report\n\n"
            "[[Casimir effect]] ↔ [[van der Waals forces]]\n",
            "utf-8",
        )

        resp = client.get("/api/reports/report_20240101_120000")
        assert resp.status_code == 200
        body = resp.json()
        assert "[[Casimir effect]]" in body["content"]

    def test_list_reports(self, client: TestClient, tmp_path: Path) -> None:
        vault = tmp_path / "wiki"
        reports_dir = vault / "04-reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for name in ["report_20240101_000000.md", "report_20240102_000000.md"]:
            (reports_dir / name).write_text("# Report\n", "utf-8")

        resp = client.get("/api/reports")
        assert resp.status_code == 200
        reports = resp.json()
        assert len(reports) == 2
        # newest first
        assert reports[0]["filename"] > reports[1]["filename"]
