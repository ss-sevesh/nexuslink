from pathlib import Path
import yaml
import hashlib
import re
import json
from loguru import logger

from nexuslink.wiki.vault.models import PaperNote, ConceptNote, HypothesisNote, VaultStats


class VaultReader:
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        hashes_path = vault_path / ".cache" / "note_hashes.json"
        if hashes_path.exists():
            try:
                self.note_hashes: dict[str, str] = json.loads(hashes_path.read_text(encoding="utf-8"))
            except Exception:
                self.note_hashes = {}
        else:
            self.note_hashes = {}

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                    return fm, parts[2]
                except Exception:
                    return {}, content
        return {}, content

    def _extract_section(self, content: str, section: str) -> str:
        pattern = rf"##\s+{re.escape(section)}\s*\n(.*?)(?=\n##|\Z)"
        m = re.search(pattern, content, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_wikilinks(self, text: str) -> list[str]:
        return re.findall(r"\[\[(.+?)\]\]", text)

    def read_all_papers(self) -> list[PaperNote]:
        papers_dir = self.vault_path / "01-papers"
        notes = []
        for md_file in sorted(papers_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                fm, body = self._parse_frontmatter(content)

                entities_section = self._extract_section(body, "Entities")
                entities = re.findall(r"-\s+\[\[(.+?)\]\]", entities_section)

                human_edited = self.detect_human_edits(md_file)

                notes.append(PaperNote(
                    path=md_file,
                    title=fm.get("title", ""),
                    authors=fm.get("authors", []) or [],
                    doi=fm.get("doi", ""),
                    domain=fm.get("domain", "") if isinstance(fm.get("domain"), str) else (fm.get("domain") or [""])[0] if fm.get("domain") else "",
                    year=fm.get("year", 0) or 0,
                    tags=fm.get("tags", []) or [],
                    entities=entities,
                    human_edited=human_edited,
                    last_pipeline_hash=self.note_hashes.get(str(md_file), ""),
                    raw_content=content,
                ))
            except Exception as e:
                logger.warning(f"Failed to read paper {md_file}: {e}")
        return notes

    def read_all_concepts(self) -> list[ConceptNote]:
        concepts_dir = self.vault_path / "02-concepts"
        notes = []
        for md_file in sorted(concepts_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                fm, body = self._parse_frontmatter(content)

                all_links = self._extract_wikilinks(body)

                appears_section = self._extract_section(body, "Appears In")
                paper_mentions = re.findall(r"-\s+\[\[(.+?)\]\]", appears_section)

                bridge_links = [l for l in all_links if l not in paper_mentions]
                linked_concepts = [l for l in all_links if l not in paper_mentions and l not in bridge_links]

                human_edited = self.detect_human_edits(md_file)

                notes.append(ConceptNote(
                    path=md_file,
                    name=fm.get("name", md_file.stem),
                    entity_type=fm.get("entity_type", ""),
                    domains=fm.get("domains", []) or [],
                    linked_concepts=linked_concepts,
                    bridge_links=bridge_links,
                    paper_mentions=paper_mentions,
                    human_edited=human_edited,
                    last_pipeline_hash=self.note_hashes.get(str(md_file), ""),
                    raw_content=content,
                ))
            except Exception as e:
                logger.warning(f"Failed to read concept {md_file}: {e}")
        return notes

    def read_all_hypotheses(self) -> list[HypothesisNote]:
        hyp_dir = self.vault_path / "03-hypotheses"
        notes = []
        for md_file in sorted(hyp_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                fm, body = self._parse_frontmatter(content)

                statement = self._extract_section(body, "Hypothesis Statement")
                human_edited = self.detect_human_edits(md_file)

                notes.append(HypothesisNote(
                    path=md_file,
                    id=fm.get("id", ""),
                    status=fm.get("status", "generated"),
                    confidence=float(fm.get("confidence", 0.0) or 0.0),
                    novelty_score=float(fm.get("novelty_score", 0.0) or 0.0),
                    feasibility_score=float(fm.get("feasibility_score", 0.0) or 0.0),
                    impact_score=float(fm.get("impact_score", 0.0) or 0.0),
                    mechanistic_depth=float(fm.get("mechanistic_depth", 0.0) or 0.0),
                    falsifiability_score=float(fm.get("falsifiability_score", 0.0) or 0.0),
                    composite_score=float(fm.get("composite_score", 0.0) or 0.0),
                    domains_spanned=fm.get("domains_spanned", []) or [],
                    cycle_generated=int(fm.get("cycle_generated", 0) or 0),
                    human_edited=human_edited,
                    last_pipeline_hash=self.note_hashes.get(str(md_file), ""),
                    statement=statement,
                    raw_content=content,
                ))
            except Exception as e:
                logger.warning(f"Failed to read hypothesis {md_file}: {e}")
        return notes

    def get_vault_stats(self) -> VaultStats:
        papers = self.read_all_papers()
        concepts = self.read_all_concepts()
        hypotheses = self.read_all_hypotheses()

        reports_dir = self.vault_path / "04-reports"
        total_reports = len(list(reports_dir.glob("*.md"))) if reports_dir.exists() else 0

        cycles_dir = self.vault_path / "05-cycles"
        total_cycles = len(list(cycles_dir.glob("*.md"))) if cycles_dir.exists() else 0

        status_counts: dict[str, int] = {}
        for h in hypotheses:
            status_counts[h.status] = status_counts.get(h.status, 0) + 1

        domains = list({p.domain for p in papers if p.domain})

        broken = self.get_broken_links()
        orphans = self.get_orphan_notes()

        return VaultStats(
            total_papers=len(papers),
            total_concepts=len(concepts),
            total_hypotheses=len(hypotheses),
            total_reports=total_reports,
            total_cycles=total_cycles,
            hypotheses_by_status=status_counts,
            domains_covered=domains,
            broken_links=len(broken),
            orphan_notes=len(orphans),
        )

    def get_notes_by_status(self, status: str) -> list[HypothesisNote]:
        return [h for h in self.read_all_hypotheses() if h.status == status]

    def get_broken_links(self) -> list[tuple[str, str]]:
        broken = []
        all_md = list(self.vault_path.rglob("*.md"))

        valid_stems = set()
        for d in ("01-papers", "02-concepts", "03-hypotheses"):
            d_path = self.vault_path / d
            if d_path.exists():
                for f in d_path.glob("*.md"):
                    valid_stems.add(f.stem.lower())

        for md_file in all_md:
            try:
                content = md_file.read_text(encoding="utf-8")
                links = self._extract_wikilinks(content)
                for link in links:
                    target = link.split("|")[0].strip()
                    if target.lower() not in valid_stems:
                        broken.append((str(md_file), target))
            except Exception as e:
                logger.warning(f"Failed to scan {md_file}: {e}")
        return broken

    def get_orphan_notes(self) -> list[Path]:
        all_md = list(self.vault_path.rglob("*.md"))

        referenced: set[str] = set()
        for md_file in all_md:
            try:
                content = md_file.read_text(encoding="utf-8")
                for link in self._extract_wikilinks(content):
                    referenced.add(link.split("|")[0].strip().lower())
            except Exception:
                pass

        orphans = []
        for d in ("01-papers", "02-concepts", "03-hypotheses"):
            d_path = self.vault_path / d
            if d_path.exists():
                for f in d_path.glob("*.md"):
                    if f.stem.lower() not in referenced:
                        orphans.append(f)
        return orphans

    def detect_human_edits(self, note_path: Path) -> bool:
        try:
            content = note_path.read_text(encoding="utf-8")
            current_hash = hashlib.sha256(content.encode()).hexdigest()
            stored = self.note_hashes.get(str(note_path), "")
            return current_hash != stored
        except Exception:
            return False

    def update_hash(self, note_path: Path):
        try:
            content = note_path.read_text(encoding="utf-8")
            self.note_hashes[str(note_path)] = hashlib.sha256(content.encode()).hexdigest()
            hashes_path = self.vault_path / ".cache" / "note_hashes.json"
            hashes_path.parent.mkdir(parents=True, exist_ok=True)
            hashes_path.write_text(json.dumps(self.note_hashes, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to update hash for {note_path}: {e}")
