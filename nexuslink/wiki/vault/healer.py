from pathlib import Path
import json
import re
import difflib
import shutil
from datetime import datetime, timezone
from loguru import logger

from nexuslink.wiki.vault.reader import VaultReader
from nexuslink.wiki.vault.models import HealReport

try:
    from nexuslink.wiki.linker.embedder import ConceptEmbedder
    _EMBEDDER_AVAILABLE = True
except ImportError:
    _EMBEDDER_AVAILABLE = False


class VaultHealer:
    def __init__(self, reader: VaultReader, embedder=None):
        self.reader = reader
        self.embedder = embedder

    def _normalize(self, name: str) -> str:
        return re.sub(r"[\s\-_]+", "", name.lower())

    def _count_backlinks(self, note_stem: str) -> int:
        count = 0
        for md_file in self.reader.vault_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                count += len(re.findall(rf"\[\[{re.escape(note_stem)}(?:\|[^\]]+)?\]\]", content))
            except Exception:
                pass
        return count

    def _replace_wikilink_in_vault(self, old_name: str, new_name: str):
        for md_file in self.reader.vault_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                updated = re.sub(
                    rf"\[\[{re.escape(old_name)}(?:\|[^\]]+)?\]\]",
                    f"[[{new_name}]]",
                    content,
                )
                if updated != content:
                    md_file.write_text(updated, encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to update wikilinks in {md_file}: {e}")

    def merge_duplicate_concepts(self) -> list[dict]:
        concepts = self.reader.read_all_concepts()
        merged_results = []
        merged_stems: set[str] = set()

        merge_history_path = self.reader.vault_path / ".cache" / "merge_history.json"
        merge_history_path.parent.mkdir(parents=True, exist_ok=True)
        if merge_history_path.exists():
            try:
                history = json.loads(merge_history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        else:
            history = []

        archive_dir = self.reader.vault_path / ".archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        for i, c1 in enumerate(concepts):
            if c1.path.stem in merged_stems:
                continue
            for c2 in concepts[i + 1:]:
                if c2.path.stem in merged_stems:
                    continue

                is_duplicate = self._normalize(c1.name) == self._normalize(c2.name)

                if not is_duplicate and self.embedder:
                    try:
                        v1 = self.embedder.embed(c1.name)
                        v2 = self.embedder.embed(c2.name)
                        dot = sum(a * b for a, b in zip(v1, v2))
                        mag1 = sum(a * a for a in v1) ** 0.5
                        mag2 = sum(b * b for b in v2) ** 0.5
                        sim = dot / (mag1 * mag2) if mag1 and mag2 else 0.0
                        if sim > 0.90:
                            is_duplicate = True
                    except Exception:
                        pass

                if is_duplicate:
                    bl1 = self._count_backlinks(c1.path.stem)
                    bl2 = self._count_backlinks(c2.path.stem)
                    keep, drop = (c1, c2) if bl1 >= bl2 else (c2, c1)

                    keep_content = keep.path.read_text(encoding="utf-8")
                    drop_content = drop.path.read_text(encoding="utf-8")
                    keep.path.write_text(
                        keep_content.rstrip() + "\n\n<!-- merged from: " + drop.path.stem + " -->\n" + drop_content,
                        encoding="utf-8",
                    )

                    archive_dest = archive_dir / drop.path.name
                    shutil.move(str(drop.path), str(archive_dest))
                    self._replace_wikilink_in_vault(drop.path.stem, keep.path.stem)

                    record = {"kept": keep.path.stem, "merged": drop.path.stem, "reason": "normalized name match"}
                    merged_results.append(record)
                    history.append(record)
                    merged_stems.add(drop.path.stem)
                    logger.info(f"Merged duplicate: {drop.path.stem} → {keep.path.stem}")

        merge_history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return merged_results

    def fix_broken_links(self) -> int:
        broken = self.reader.get_broken_links()
        concepts_dir = self.reader.vault_path / "02-concepts"
        concept_stems = [f.stem for f in concepts_dir.glob("*.md")] if concepts_dir.exists() else []
        fixes = 0

        for source_path_str, broken_target in broken:
            source_path = Path(source_path_str)
            matches = difflib.get_close_matches(broken_target, concept_stems, n=1, cutoff=0.85)
            if matches:
                correct = matches[0]
                try:
                    content = source_path.read_text(encoding="utf-8")
                    updated = re.sub(
                        rf"\[\[{re.escape(broken_target)}(?:\|[^\]]+)?\]\]",
                        f"[[{correct}]]",
                        content,
                    )
                    if updated != content:
                        source_path.write_text(updated, encoding="utf-8")
                        fixes += 1
                        logger.info(f"Fixed link: [[{broken_target}]] → [[{correct}]] in {source_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to fix link in {source_path}: {e}")
            else:
                stub_path = concepts_dir / f"{broken_target}.md"
                if not stub_path.exists():
                    try:
                        stub_path.write_text(
                            f"---\nstatus: stub\n---\n\nStub concept — auto-created. Add context or delete.\n",
                            encoding="utf-8",
                        )
                        fixes += 1
                        logger.info(f"Created stub: {stub_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to create stub {stub_path}: {e}")

        return fixes

    def prune_low_quality_concepts(self) -> list[Path]:
        concepts = self.reader.read_all_concepts()
        archive_dir = self.reader.vault_path / ".archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        pruned = []

        for c in concepts:
            if len(c.paper_mentions) <= 1 and len(c.bridge_links) == 0 and len(c.name) < 4:
                dest = archive_dir / c.path.name
                shutil.move(str(c.path), str(dest))
                pruned.append(c.path)
                logger.info(f"Pruned low-quality concept: {c.path.name}")

        return pruned

    def update_stale_notes(self) -> int:
        papers = self.reader.read_all_papers()
        concepts_dir = self.reader.vault_path / "02-concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)
        existing_stems = {f.stem.lower() for f in concepts_dir.glob("*.md")}
        count = 0

        for paper in papers:
            if not paper.human_edited:
                continue
            content = paper.path.read_text(encoding="utf-8")
            _, body = self.reader._parse_frontmatter(content)
            entities_section = self.reader._extract_section(body, "Entities")
            entities = re.findall(r"-\s+\[\[(.+?)\]\]", entities_section)

            for entity in entities:
                if entity.lower() not in existing_stems:
                    stub_path = concepts_dir / f"{entity}.md"
                    stub_path.write_text(
                        f"---\nname: {entity}\n---\n\nAuto-created from human-edited paper: {paper.path.stem}.\n",
                        encoding="utf-8",
                    )
                    existing_stems.add(entity.lower())
                    count += 1
                    logger.info(f"Created concept from stale note: {entity}")

        return count

    def heal(self, dry_run: bool = False) -> HealReport:
        if dry_run:
            logger.info("Dry run — no files will be written")
            merged = self.merge_duplicate_concepts() if False else []
            fixed = 0
            pruned = []
            updated = 0
        else:
            merged = self.merge_duplicate_concepts()
            fixed = self.fix_broken_links()
            pruned = self.prune_low_quality_concepts()
            updated = self.update_stale_notes()

        report = HealReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            duplicates_merged=len(merged),
            links_fixed=fixed,
            concepts_pruned=len(pruned),
            notes_updated=updated,
        )

        history_path = self.reader.vault_path / ".cache" / "heal_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        else:
            history = []

        history.append(report.model_dump())
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        return report
