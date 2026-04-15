"""Cyclical pipeline — the core differentiator vs one-shot systems."""

import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from nexuslink.wiki.vault.reader import VaultReader
from nexuslink.wiki.vault.healer import VaultHealer
from nexuslink.wiki.vault.feedback import FeedbackLoop
from nexuslink.wiki.vault.expander import AutonomousExpander
from nexuslink.wiki.vault.models import CycleReport


class NexusLinkCycle:
    def __init__(self, vault_path: Path, config=None):
        self.vault_path = Path(vault_path)
        self.config = config
        self.cache_dir = self.vault_path / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cycle_history_path = self.cache_dir / "cycle_history.json"
        self.reader = VaultReader(self.vault_path)

    def _load_cycle_history(self) -> list[dict]:
        if self.cycle_history_path.exists():
            try:
                return json.loads(self.cycle_history_path.read_text())
            except Exception:
                return []
        return []

    def _save_cycle_history(self, history: list[dict]):
        self.cycle_history_path.write_text(json.dumps(history, indent=2))

    def _get_next_cycle_number(self) -> int:
        history = self._load_cycle_history()
        if history:
            return max(h.get("cycle_number", 0) for h in history) + 1
        return 1

    def run_cycle(self, cycle_number: int | None = None) -> CycleReport:
        if cycle_number is None:
            cycle_number = self._get_next_cycle_number()

        timestamp = datetime.now().isoformat()
        logger.info(f"=== CYCLE {cycle_number} START ===")

        # Step 1 — READ
        logger.info("Step 1: Reading vault state...")
        stats_before = self.reader.get_vault_stats()

        # Step 2 — HEAL
        logger.info("Step 2: Healing vault...")
        healer = VaultHealer(self.reader)
        heal_report = healer.heal(dry_run=False)
        logger.info(f"Healed: {heal_report.duplicates_merged} merges, {heal_report.links_fixed} link fixes, {heal_report.concepts_pruned} pruned")

        # Step 3 — FEEDBACK
        logger.info("Step 3: Collecting feedback...")
        feedback = FeedbackLoop(self.reader)
        few_shot = feedback.build_few_shot_examples()
        calibration = feedback.calibrate_scoring()
        rejected_pairs = feedback.get_rejected_bridge_pairs()
        concept_overrides = feedback.get_human_concept_overrides()
        logger.info(f"Feedback: {len(few_shot.positive_examples)} positive, {len(few_shot.negative_examples)} negative, {len(rejected_pairs)} rejected pairs")

        # Step 4 — LINK
        logger.info("Step 4: Running linker...")
        hypotheses_generated = 0
        try:
            from nexuslink.wiki.linker.pipeline import run_linking
            import asyncio
            asyncio.run(run_linking())
        except Exception as e:
            logger.warning(f"Linker step failed: {e}")

        # Step 5 — HYPOTHESIZE
        logger.info("Step 5: Generating hypotheses...")
        try:
            from nexuslink.llm.hypothesis.pipeline import run_hypothesis_pipeline
            from nexuslink.config import NexusConfig
            import asyncio
            _cfg = self.config or NexusConfig()
            result = asyncio.run(run_hypothesis_pipeline(vault_path=self.vault_path, config=_cfg))
            hypotheses_generated = result.get("hypotheses_generated", 0) if isinstance(result, dict) else 0
        except Exception as e:
            logger.warning(f"Hypothesis step failed: {e}")

        # Step 5.5 — INTEGRITY CHECK
        logger.info("Step 5.5: Checking evidence integrity...")
        integrity_flags = 0
        try:
            from nexuslink.wiki.vault.integrity import EvidenceIntegrityChecker
            checker = EvidenceIntegrityChecker(self.vault_path)
            integrity_results = checker.check_all_hypotheses()
            integrity_flags = sum(1 for r in integrity_results if r.retraction_flags)
            logger.info(f"Integrity: {len(integrity_results)} checked, {integrity_flags} flagged")
        except Exception as e:
            logger.warning(f"Integrity check failed: {e}")

        # Step 6 — EXPAND
        logger.info("Step 6: Expanding vault...")
        papers_expanded = 0
        try:
            s2_key = None
            if self.config and hasattr(self.config, "semantic_scholar_api_key"):
                s2_key = self.config.semantic_scholar_api_key
            expander = AutonomousExpander(self.reader, api_key=s2_key)
            expand_result = expander.auto_expand_cycle(max_new_papers=10)
            papers_expanded = expand_result.get("papers_ingested", 0)
        except Exception as e:
            logger.warning(f"Expansion step failed: {e}")

        # Step 7 — RECORD
        logger.info("Step 7: Recording cycle...")
        self.reader = VaultReader(self.vault_path)  # re-read after changes
        stats_after = self.reader.get_vault_stats()

        report = CycleReport(
            cycle_number=cycle_number,
            timestamp=timestamp,
            papers_before=stats_before.total_papers,
            papers_after=stats_after.total_papers,
            concepts_before=stats_before.total_concepts,
            concepts_after=stats_after.total_concepts,
            hypotheses_generated=hypotheses_generated,
            papers_auto_expanded=papers_expanded,
            heal_actions=heal_report.duplicates_merged + heal_report.links_fixed + heal_report.concepts_pruned,
            feedback_applied=bool(few_shot.positive_examples or few_shot.negative_examples),
        )

        history = self._load_cycle_history()
        history.append(report.model_dump())
        self._save_cycle_history(history)

        # Step 8 — LOG
        logger.info("Step 8: Writing cycle note...")
        self._write_cycle_note(report)

        logger.info(f"=== CYCLE {cycle_number} DONE === Papers: {report.papers_before}→{report.papers_after}, Hypotheses: {report.hypotheses_generated}, Integrity flags: {integrity_flags}")
        return report

    def _write_cycle_note(self, report: CycleReport):
        cycles_dir = self.vault_path / "05-cycles"
        cycles_dir.mkdir(parents=True, exist_ok=True)
        note_path = cycles_dir / f"cycle_{report.cycle_number}.md"
        content = f"""---
cycle_number: {report.cycle_number}
timestamp: "{report.timestamp}"
papers_added: {report.papers_after - report.papers_before}
concepts_added: {report.concepts_after - report.concepts_before}
hypotheses_generated: {report.hypotheses_generated}
heal_actions: {report.heal_actions}
papers_auto_expanded: {report.papers_auto_expanded}
feedback_applied: {report.feedback_applied}
---

## Cycle {report.cycle_number} Summary

- Papers: {report.papers_before} → {report.papers_after} (+{report.papers_after - report.papers_before})
- Concepts: {report.concepts_before} → {report.concepts_after} (+{report.concepts_after - report.concepts_before})
- Hypotheses generated: {report.hypotheses_generated}
- Heal actions: {report.heal_actions}
- Papers auto-expanded: {report.papers_auto_expanded}
- Feedback applied: {report.feedback_applied}
"""
        note_path.write_text(content, encoding="utf-8")

    def run_continuous(self, interval_minutes: int = 60, max_cycles: int = 10):
        import time
        no_new_bridges_count = 0

        for i in range(max_cycles):
            report = self.run_cycle()

            if report.papers_after == report.papers_before and report.hypotheses_generated == 0:
                no_new_bridges_count += 1
                if no_new_bridges_count >= 2:
                    logger.info("Vault stabilized — no new content for 2 cycles. Stopping.")
                    break
            else:
                no_new_bridges_count = 0

            if i < max_cycles - 1:
                logger.info(f"Sleeping {interval_minutes} minutes before next cycle...")
                time.sleep(interval_minutes * 60)
