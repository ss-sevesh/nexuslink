# NexusLink Progress

## Phase 1: Core Pipeline DONE
- [x] PDF/ArXiv/DOI ingestion
- [x] Entity extraction (spaCy + phrase patterns)
- [x] Cross-domain bridge finding
- [x] Hypothesis generation + scoring
- [x] Obsidian vault output (wiki/01-papers/, wiki/02-concepts/, wiki/03-hypotheses/)

## Phase 2: Self-Refining System DONE
- [x] VaultReader (hash-based human edit detection)
- [x] VaultHealer (merge, fix links, prune, propagate edits)
- [x] FeedbackLoop (few-shot examples, scoring calibration, rejected bridge pairs)
- [x] AutonomousExpander (Semantic Scholar search, auto-ingest)
- [x] NexusLinkCycle (8-step cyclical pipeline)
- [x] EvidenceIntegrityChecker (retraction via CrossRef, reliability scoring via S2)
- [x] Benchmark module (one-shot vs cyclical comparison, LaTeX export)
- [x] vault/__init__.py (all classes exported)
- [x] All path references updated to numbered dirs (01-papers, 02-concepts)
- [x] All file I/O uses encoding="utf-8" (Windows safe)
- [x] Templates: Hypothesis, Concept, Paper, Cycle all updated

## Phase 3: Validation (CURRENT)
- [ ] Feed 15 papers: 5 neuroscience + 5 materials science + 5 ecology
- [ ] Run 3 cycles, collect benchmark data
- [ ] Screenshot Obsidian graph view for paper
- [ ] Find professor co-author

## Phase 4: Publication
- [ ] Paper draft: "NexusLink: Self-Refining Cross-Domain Hypothesis Generation with Evidence Integrity"
- [ ] Target: NeurIPS AI4Science Workshop / AAAI
- [ ] Demo video

## Phase 5: Funding
- [ ] NSF Convergence Accelerator proposal
- [ ] SERB CRG proposal (if applicable)
