"""Domain classifier: maps a RawDocument to ranked scientific domains."""

from __future__ import annotations

from nexuslink.raw.schemas.models import RawDocument

# ---------------------------------------------------------------------------
# Domain keyword lexicons — roughly 15-20 discriminating terms per domain
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "physics": [
        "quantum", "photon", "electron", "particle", "wave", "field",
        "entropy", "thermodynamics", "casimir", "superconductor", "plasma",
        "relativity", "boson", "fermion", "spin", "hadron", "neutrino",
        "condensate", "phonon", "magnetism",
    ],
    "chemistry": [
        "molecule", "reaction", "catalyst", "synthesis", "bond", "element",
        "compound", "polymer", "spectroscopy", "solvent", "reagent",
        "oxidation", "reduction", "equilibrium", "stoichiometry", "ligand",
        "chelation", "isomer", "titration", "electrolysis",
    ],
    "biology": [
        "gene", "protein", "cell", "organism", "evolution", "metabolism",
        "enzyme", "dna", "rna", "genome", "chromosome", "membrane",
        "transcription", "translation", "mitosis", "ribosome", "nucleus",
        "photosynthesis", "ecology", "biodiversity",
    ],
    "medicine": [
        "clinical", "patient", "therapy", "disease", "drug", "treatment",
        "diagnosis", "pathology", "trial", "symptom", "prognosis", "dose",
        "antibody", "vaccine", "surgery", "oncology", "pharmacology",
        "epidemiology", "biomarker", "randomized",
    ],
    "cs": [
        "algorithm", "neural", "network", "learning", "data", "computing",
        "optimization", "graph", "complexity", "compiler", "cryptography",
        "heuristic", "distributed", "latency", "bandwidth", "cache",
        "transformer", "gradient", "inference", "architecture",
    ],
    "materials_science": [
        "material", "alloy", "crystal", "composite", "nanostructure",
        "mechanical", "thermal", "surface", "deposition", "hardness",
        "elasticity", "fracture", "corrosion", "semiconductor", "ceramic",
        "thin film", "grain", "microstructure", "diffusion", "annealing",
    ],
    "engineering": [
        "design", "system", "circuit", "sensor", "actuator", "control",
        "fabrication", "simulation", "manufacturing", "reliability",
        "efficiency", "prototype", "embedded", "signal", "impedance",
        "torque", "bearing", "hydraulic", "pneumatic", "calibration",
    ],
    "mathematics": [
        "theorem", "proof", "equation", "topology", "algebra", "calculus",
        "manifold", "probability", "stochastic", "combinatorics", "matrix",
        "eigenvalue", "integral", "differential", "group", "field",
        "homotopy", "convergence", "sequence", "bijection",
    ],
}

# ArXiv category prefixes → domain (longest prefix wins)
_ARXIV_PREFIX_MAP: list[tuple[str, str]] = sorted(
    [
        ("cond-mat", "physics"),
        ("quant-ph", "physics"),
        ("hep-", "physics"),
        ("astro-ph", "physics"),
        ("physics.", "physics"),
        ("q-bio", "biology"),
        ("q-fin", "mathematics"),
        ("stat", "mathematics"),
        ("math.", "mathematics"),
        ("math-ph", "mathematics"),
        ("cs.", "cs"),
        ("eess.", "engineering"),
        ("econ.", "mathematics"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,  # longest prefix first
)


def classify_domain(doc: RawDocument) -> list[tuple[str, float]]:
    """Return domains ranked by confidence score ∈ (0, 1].

    Uses keyword matching over abstract + domain_tags, boosted by ArXiv
    category prefixes when present.  Only domains with a non-zero score
    are returned.
    """
    text = " ".join([
        doc.abstract,
        doc.title,
        " ".join(doc.domain_tags),
    ]).lower()

    raw_scores: dict[str, float] = {}

    # Keyword score: fraction of domain keywords found in text
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        raw_scores[domain] = hits / len(keywords)

    # ArXiv category boost: +0.4 per matching prefix (capped at 1.0 later)
    for tag in doc.domain_tags:
        for prefix, domain in _ARXIV_PREFIX_MAP:
            if tag.startswith(prefix):
                raw_scores[domain] = raw_scores.get(domain, 0.0) + 0.40
                break  # one boost per tag

    total = sum(raw_scores.values()) or 1.0
    ranked = sorted(
        [(d, round(s / total, 4)) for d, s in raw_scores.items() if s > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked
