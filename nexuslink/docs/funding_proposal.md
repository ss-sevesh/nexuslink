# NexusLink Funding Proposal

## Project Title
NexusLink: An AI-Driven Cross-Domain Knowledge Graph for Automated Scientific Hypothesis Generation

## One-Sentence Summary
NexusLink builds a machine-readable bridge between scientific domains and generates novel, falsifiable research hypotheses that no domain expert would produce alone.

## Problem Statement

Scientific literature grows at 4+ million papers per year. Researchers spend 30–50% of their time on literature review and still miss relevant work outside their immediate subfield. The most transformative discoveries in science — CRISPR from bacterial immunity, Velcro from burr hooks, penicillin from a contaminated petri dish — come from accidental cross-domain observation.

NexusLink makes this systematic.

Existing tools (Semantic Scholar, ResearchRabbit, Elicit, Consensus) perform retrieval and summarisation within a domain. None build cross-domain knowledge graphs and generate hypotheses from the conceptual bridges they find. This is the gap.

## Proposed Solution

A three-layer pipeline:

1. **RAW**: Ingest papers from any domain. Extract entities, claims, and methodology sections. Embed into a shared semantic vector space.

2. **WIKI**: Construct a knowledge graph where nodes are scientific concepts and edges are typed relations (cites, contradicts, extends, analogous_to, enables). A dedicated linker finds non-obvious bridges between entities from different domains — cases where embedding similarity is high but citation overlap is zero.

3. **LLM**: For each cross-domain bridge, generate a structured hypothesis: *"If [finding from domain A] and [mechanism from domain B], then [novel prediction for domain C]."* Each hypothesis is scored for novelty, feasibility, and potential impact, and validated against known literature. Output is a publication-ready report.

## Target Funding Programs

### NSF Convergence Accelerator — Track H: AI for Science
- Alignment: NexusLink directly addresses convergence of AI and scientific discovery
- Ask: $750K Phase I (proof of concept with 3 domains)
- Deliverable: Open-source tool + peer-reviewed methodology paper

### DARPA AIE (AI Exploration)
- Alignment: Exploratory AI research with high risk/reward profile
- Ask: $1.5M over 18 months
- Deliverable: System capable of generating verifiable novel hypotheses in biology/materials/physics

### Wellcome Trust Discovery Award
- Alignment: Biomedical knowledge synthesis and AI tools for researchers
- Ask: £500K
- Deliverable: Biomedical hypothesis engine with clinical trial suggestion capability

### Sloan Research Fellowship — Tools Track
- Alignment: Foundational tooling for the scientific community
- Ask: Fellowship nomination (investigator-initiated)
- Deliverable: Open-source release, reproducibility benchmark, and methodology publication

## Preliminary Results

*[To be filled in as Phase 1 is completed.]*

- Ingestion pipeline: X papers/min
- Entity extraction F1: X% (vs. gold standard)
- Cross-domain bridge precision: X% (human expert evaluation)
- Hypothesis novelty rate: X% (not found in Semantic Scholar)

## Team
*[To be filled in.]*

## Timeline

| Quarter | Milestone |
|---|---|
| Q1 | RAW layer complete; 10K papers ingested across 3 domains |
| Q2 | WIKI layer complete; knowledge graph queryable |
| Q3 | LLM layer complete; first hypothesis batch generated |
| Q4 | Expert evaluation; paper submitted; public demo |

## Open Science Commitment
All code, models, and datasets produced under this grant will be released under MIT/CC-BY licences. The hypothesis benchmark (human-evaluated cross-domain novelty) will be contributed to the community as a shared evaluation resource.
