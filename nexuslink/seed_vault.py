"""Seed the NexusLink vault with 15 hardcoded papers across 3 domains."""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on path so 'nexuslink' resolves correctly
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

PAPERS = [
    # Remaining (9 already ingested, skipping 2306.08478 — too slow)
    "2309.15891",  # mycorrhizal networks
    "2305.08561",  # swarm intelligence
    "2307.12890",  # coral reef climate
    "2304.11247",  # invasive species cascades
    "2308.09643",  # ecosystem tipping points
]


async def main():
    from raw.ingestion.pipeline import run_ingestion
    from wiki.linker.pipeline import run_linking
    from wiki.vault.reader import VaultReader

    print("=" * 60)
    print("NexusLink Vault Seeder — 15 papers, 3 domains")
    print("=" * 60)

    # Step 1 — Ingest
    ingested = 0
    failed = 0
    for arxiv_id in PAPERS:
        print(f"\n  Ingesting {arxiv_id}...")
        try:
            result = await run_ingestion(arxiv_id)
            ingested += 1
            entities = result.get("entities_found", 0) if isinstance(result, dict) else "?"
            title = result.get("title", "") if isinstance(result, dict) else ""
            print(f"  OK — {title[:65]} ({entities} entities)")
        except Exception as e:
            failed += 1
            print(f"  FAIL — {e}")

    print(f"\nIngestion: {ingested} OK, {failed} failed")
    print("=" * 60)

    # Step 2 — Link
    print("\nRunning cross-domain linking...")
    try:
        link_stats = await run_linking()
        print(f"  Papers processed : {link_stats.get('papers_processed', 0)}")
        print(f"  Concepts         : {link_stats.get('total_concepts', 0)}")
        print(f"  Bridges          : {link_stats.get('total_bridges', 0)}")
        print(f"  Domains          : {link_stats.get('domains_covered', 0)}")
        print(f"  Notes written    : {link_stats.get('concept_notes_written', 0)}")
    except Exception as e:
        print(f"  Linking failed: {e}")
    print("=" * 60)

    # Step 3 — Vault stats
    print("\nVault stats:")
    try:
        reader = VaultReader(Path("wiki"))
        stats = reader.get_vault_stats()
        print(f"  Papers     : {stats.total_papers}")
        print(f"  Concepts   : {stats.total_concepts}")
        print(f"  Hypotheses : {stats.total_hypotheses}")
        print(f"  Domains    : {stats.domains_covered}")
    except Exception as e:
        print(f"  Stats failed: {e}")
    print("=" * 60)

    print(f"\nSEED COMPLETE — {ingested}/15 ingested")
    print("  Next: uv run nexuslink cycle")
    print("  Then: uv run nexuslink integrity")


if __name__ == "__main__":
    asyncio.run(main())
