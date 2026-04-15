"""Named entity recognition over raw scientific text using spaCy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..schemas.models import ExtractedEntity, RawDocument

if TYPE_CHECKING:
    import spacy.language

# ---------------------------------------------------------------------------
# spaCy label → our entity type
# ---------------------------------------------------------------------------
_LABEL_MAP: dict[str, str] = {
    # sciSpaCy
    "CHEMICAL": "chemical",
    "SIMPLE_CHEMICAL": "chemical",
    "GENE": "gene",
    "GENE_OR_GENE_PRODUCT": "gene",
    "PROTEIN": "gene",
    "CELL_TYPE": "organism",
    "ORGANISM": "organism",
    "TAXON": "organism",
    "DISEASE": "phenomenon",
    "PHENOTYPE": "phenomenon",
    # en_core_web_sm
    "PRODUCT": "material",
    # ORG is included because en_core_web_sm labels most ML/NLP concepts
    # (BLEU, WMT, GPT-3, Transformer variants) as ORG.
    # Noise is removed by _is_valid_entity and _BLOCKLIST below.
    "ORG": "phenomenon",
}

# ---------------------------------------------------------------------------
# Phrase patterns — surface forms that NER misses
# Keys = our entity_type; values = canonical surface forms (case-insensitive match)
# ---------------------------------------------------------------------------
_PHRASE_PATTERNS: dict[str, list[str]] = {
    "method": [
        # ── ML / NLP ──────────────────────────────────────────────────────
        "deep learning", "machine learning", "transfer learning",
        "gradient descent", "backpropagation", "fine-tuning",
        "pre-training", "self-supervised learning", "reinforcement learning",
        "few-shot learning", "zero-shot learning", "in-context learning",
        "prompt tuning", "instruction tuning", "contrastive learning",
        "knowledge distillation", "model pruning", "quantization",
        # Architecture components
        "transformer", "self-attention", "multi-head attention",
        "cross-attention", "attention mechanism", "feed-forward network",
        "layer normalization", "batch normalization", "residual connection",
        "positional encoding", "position encoding", "encoder-decoder",
        "mixture of experts", "sparse routing", "top-k routing",
        "convolutional neural network", "recurrent neural network",
        "long short-term memory", "graph neural network",
        "variational autoencoder", "generative adversarial network",
        "diffusion model", "score matching",
        # ── Biology / Medicine ────────────────────────────────────────────
        "PCR", "CRISPR", "ELISA", "western blot", "RNA sequencing",
        "mass spectrometry", "flow cytometry", "X-ray crystallography",
        "cryo-EM", "molecular dynamics", "Monte Carlo",
        "gel electrophoresis", "immunofluorescence",
        "whole genome sequencing", "single-cell RNA sequencing",
        "chromatin immunoprecipitation", "fluorescence microscopy",
        # ── Chemistry ────────────────────────────────────────────────────
        "density functional theory", "DFT", "ab initio",
        "nuclear magnetic resonance", "NMR", "Raman spectroscopy",
        "infrared spectroscopy", "cyclic voltammetry",
        "high-performance liquid chromatography", "HPLC",
        # ── Physics ───────────────────────────────────────────────────────
        "quantum Monte Carlo", "density matrix renormalization group",
        "variational Monte Carlo", "quantum annealing",
        "tensor network", "mean field theory",
        # ── Data science ──────────────────────────────────────────────────
        "principal component analysis", "PCA",
        "support vector machine", "SVM", "random forest",
        "gradient boosting", "XGBoost", "SHAP",
        "Bayesian optimization", "Gaussian process",
    ],
    "material": [
        # Materials science
        "graphene", "carbon nanotube", "nanoparticle", "quantum dot",
        "polymer", "hydrogel", "aerogel", "perovskite", "zeolite",
        "metal-organic framework", "MOF", "thin film", "semiconductor",
        "topological insulator", "2D material", "heterostructure",
        "shape memory alloy", "piezoelectric material",
        # Biology materials
        "DNA", "RNA", "protein", "lipid nanoparticle", "hydrogel scaffold",
        "extracellular matrix", "viral vector",
        # Chemistry
        "catalyst", "electrolyte", "photoinitiator", "covalent bond",
        "hydrogen bond", "van der Waals complex",
    ],
    "phenomenon": [
        # Physics
        "Casimir effect", "superconductivity", "superfluidity",
        "van der Waals", "quantum entanglement", "Bose-Einstein condensate",
        "piezoelectricity", "ferromagnetism", "antiferromagnetism",
        "spin-orbit coupling", "Anderson localization",
        "quantum Hall effect", "topological order",
        # Biology
        "photosynthesis", "gecko adhesion", "apoptosis",
        "synaptic plasticity", "long-term potentiation",
        "epigenetic regulation", "horizontal gene transfer",
        "quorum sensing", "chemotaxis",
        # Chemistry
        "catalysis", "photocatalysis", "electrochemical reduction",
        "chirality", "aromaticity", "hydrogen bonding",
        # ML phenomena
        "scaling law", "emergent behavior", "catastrophic forgetting",
        "double descent", "grokking", "hallucination",
        "model collapse", "reward hacking",
        "expert capacity", "load balancing", "sparse activation",
        # Cross-domain universals
        "self-organization", "phase transition", "criticality",
        "feedback loop", "bifurcation", "attractor",
        "information bottleneck", "entropy",
    ],
}

# ---------------------------------------------------------------------------
# Blocklist — exact names (case-sensitive) that are never scientific concepts
# ---------------------------------------------------------------------------
_BLOCKLIST: frozenset[str] = frozenset({
    # Too generic
    "Model", "model", "Data", "data", "Results", "results",
    "Hardware", "hardware", "Representation", "representation",
    "Token", "token", "Tokens", "tokens", "Text", "text",
    "Figure", "figure", "Table", "table", "Section", "section",
    "Appendix", "appendix", "Algorithm", "algorithm",
    "Computational", "computational", "byte", "Byte",
    "Expert", "expert", "Dense", "dense", "Post", "post",
    "Init", "init", "Output", "output", "Input", "input",
    "Layer", "layer", "Block", "block", "Head", "head",
    "Loss", "loss", "Score", "score", "Weight", "weight",
    "Metric", "metric", "Baseline", "baseline", "Setup", "setup",
    "NMT",  # too generic abbreviation in most contexts
    # Data types / precision formats
    "float32", "bfloat16", "int8", "float16", "float64", "fp16", "fp32",
    # GPU/TPU hardware identifiers (not concepts)
    "K80", "P100", "V100", "A100", "H100", "TPU v2", "TPU v3",
    # Academic venue abbreviations — not scientific concepts
    "ACL", "NAACL", "EMNLP", "CoRR", "USENIX", "NeurIPS", "ICML",
    "ICLR", "CVPR", "ICCV", "ECCV", "AAAI", "IJCAI", "SIGIR",
    "ACM", "IEEE", "ASRU", "ASONAM", "Interspeech",
    # Tech companies / cloud platforms
    "AMD", "AWS", "Google", "Microsoft", "Apple", "Meta", "Amazon",
    "OpenAI", "Anthropic", "DeepMind", "NVIDIA",
    # Universities / government agencies (not concepts)
    "MIT", "CMU", "MSR", "NSF", "NIH", "DOE", "DARPA", "IARPA",
    # Generic acronyms that are not scientific concepts
    "API", "AIE", "ALUM", "AUC", "ASR", "ARC", "ARO",
    # Common researcher surnames captured by spaCy ORG label
    # (Pattern: single Title-Case word, not a known concept)
    # These are added per-domain as they appear — the fix is in _is_valid_entity
    "Devlin", "Shazeer", "Zoph", "Radford", "Hooker", "Jacobs",
    "Lepikhin", "Narayan", "Rosenbaum", "Sutton", "Wang", "Zellers",
    "Fedus", "Niki", "Eigen", "Sutskever", "Bengio", "Hinton",
    "LeCun", "Schmidhuber", "Vaswani", "Raffel", "Brown",
    # Generic single-letter or fragment labels
    "MTF", "MTF.",
})

# Patterns that indicate noise regardless of content
_NOISE_PATTERNS = re.compile(
    r"^Table\s+\d"                         # Table 3, Table 3.9, etc.
    r"|^Figure\s+\d"                        # Figure 1, etc.
    r"|^Section\s+\d"                       # Section 3.2, etc.
    r"|@"                                   # email addresses
    r"|^https?://"                          # URLs
    r"|^[=|`>\[<~\u201c\u2018\"\']"        # starts with markup or quote chars
    r"|\|\|"                                # double pipe
    r"|et al"                               # citation fragments
    r"|~~"                                  # strikethrough markup
    r"|^[Bb][Rr]>"                          # HTML artifact (Br>, br>)
    r"|^[^\x00-\x7F]"                       # starts with non-ASCII (↑, ×, etc.)
    r"|[×÷±∑∏∂∫√|\u2212\u00b1]"            # math operators, pipe, unicode minus/plus
    r"|^(Conference|Proceedings|Journal|Workshop|Symposium|Annual|International)\s"
                                            # academic venue names
    r"|^(ACM|IEEE|AAAI|ICLR|NeurIPS|ICML|CVPR|ECCV|ICCV|NAACL|EMNLP|ACL)\s"
                                            # venue abbreviations as prefixes
    r"|\b(University|Institute|Laboratory|College|Church|Brain Team|Research Lab)\b"
                                            # institutional names
    r"|\b(Press|Publishing|Publisher|Publishers)\b"
                                            # publisher names
    r"|\b(Conference on|Processing Systems|Technology Conference)\b"
                                            # venue/journal name fragments
    r"|,\s*(Inc\.|Ltd\.|Corp\.|LLC)"        # company suffixes
    r"|\s&\s[A-Z][a-z]+"                   # "X & Lastname" citation patterns
    r"|^[A-Z]{2,5}\d{2,4}$"               # citation codes like RNSS18, ACL19
    r"|\.\w{2,4}$"                          # file extensions / URLs (.com, .org)
    r"|^[A-Z]\.\s"                          # initial + period (e.g. "D. Switch")
    r"|\[\]"                                # square brackets (table/code artifacts)
    r"|\[[\d,\s]+\]"                        # citation markers like [1] [1,2]
    r"|\+\s"                                # arithmetic + operator
    r"|,\s*[A-Z]"                           # comma before capital (list fragments)
    r"|'\d{2}\b"                            # apostrophe+year: WMT'16
    r"|\bto\s+[A-Z][a-z]"                  # "to Tensorflow" — description fragment
    r"|\w*[^\x00-\x7F]\w*\s+[A-Z][a-z]+"  # non-ASCII word followed by surname
    r"|^arXiv"                              # arXiv prefixes
    r"|^abs/"                               # arXiv abs/ paths
    r"|^([A-Z]\.){2,}"                      # dotted abbreviations: A.G.B., U.S.A.
    r"|^[A-Z]+-[A-Z]\d+$"                  # benchmark splits: ANLI-A1, WMT-DE2
    r"|^[A-Z]\d+[a-z]*$"                    # single-letter + number codes: A11b (NOT Cas9)
    r"|^Theorem\s+\d"                       # Theorem 4.2, Theorem 1
    r"|^Lemma\s+\d"                         # Lemma 3
    r"|^Corollary\s+\d"                     # Corollary 1
    r"|^[A-Z]{8,}$"                         # pure all-caps junk >= 8 chars: COUNTERFACT
)

# Person-name patterns (various formats that appear in paper author/citation sections)
_PERSON_RE = re.compile(
    r"^[A-Z][a-z]+\s+[A-Z][a-z]+$"                     # "John Smith"
    r"|^[A-Z][a-z]+\s+[A-Z]+\s+[A-Z][a-z]+$"           # "Quoc VV Le"
    r"|^[A-Z]\.\s+[A-Z][a-z]+"                          # "J. Smith"
    r"|^[A-Z][a-z]+-[A-Z][a-z]+\s+[A-Z][a-z]+$"        # "Jean-Baptiste Cordonnier"
    r"|^[A-Z][a-z]+-[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+$"  # "Marc-Andre Foo-Bar"
    r"|^[A-Z][a-z]+[A-Z][a-z]+\s+[A-Z][a-z]+$"         # "SouYoung Jin"
    r"|^[A-Z][a-z]+\s+and$"                             # "Howard and" (author + conjunction)
    r"|^[A-Z][a-z]+-[A-Z][a-z]+$"                       # "Foo-Bar" hyphenated surnames
)

# Two-word technical terms that superficially look like person names
_KNOWN_TWO_WORD_CONCEPTS: frozenset[str] = frozenset({
    # ML / NLP methods
    "Deep Learning", "Machine Learning", "Transfer Learning",
    "Switch Transformer", "Mesh Tensorflow", "Natural Language",
    "Neural Network", "Language Model", "Sparse Transformer",
    "Few-Shot", "Zero-Shot", "Common Crawl", "Winograd Schema",
    "Byte Pair", "Attention Mechanism", "Gradient Descent",
    "Multi-Head Attention", "Self-Attention", "Cross-Attention",
    "Feed-Forward Network", "Layer Normalization", "Batch Normalization",
    "Residual Connection", "Positional Encoding", "Position Encoding",
    "Encoder-Decoder", "Mixture-of-Experts", "Long Short-Term Memory",
    "Recurrent Neural Network", "Convolutional Neural Network",
    "Graph Neural Network", "Variational Autoencoder",
    "Generative Adversarial Network", "Diffusion Model",
    "Knowledge Distillation", "Model Pruning",
    # Quantum physics / condensed matter
    "Quantum Sensing", "Quantum Entanglement", "Quantum Computing",
    "Quantum Annealing", "Spin Resonance", "Spin-Orbit Coupling",
    "Paramagnetic Defect", "Hexagonal Boron Nitride",
    "Topological Insulator", "Bose-Einstein Condensate",
    "Casimir Effect", "Anderson Localization",
    "Quantum Hall", "Quantum Dot", "Quantum Error",
    "Van der Waals", "2D Material",
    # Biology / biochemistry
    "Protein Structure", "Protein Folding", "Amino Acid",
    "Gene Expression", "Gene Regulation", "Genome Editing",
    "Cas9", "Guide RNA", "Adaptive Immunity", "Bacterial Immunity",
    "Horizontal Gene Transfer", "RNA Interference", "Alternative Splicing",
    "Epigenetic Regulation", "Synaptic Plasticity", "Hebbian Learning",
    "Dendritic Computation", "Predictive Coding", "Credit Assignment",
    "Spiking Neuron", "Neural Circuit", "Biological Neural",
    "Long-Term Potentiation", "Action Potential",
    "Molecular Dynamics", "Evolutionary Coupling", "Residue Contact",
    # Ecology / earth systems
    "Climate Change", "Biodiversity Loss", "Tipping Points",
    "Planetary Boundaries", "Feedback Loop", "Earth System",
    "Nitrogen Cycle", "Carbon Cycle", "Ecosystem Services",
    "Resilience Theory",
    # Chemistry / materials
    "Hydrogen Bond", "Covalent Bond", "Ionic Bond",
    "Reaction Kinetics", "Catalytic Mechanism",
    # Cross-domain
    "Phase Transition", "Scaling Law", "Information Bottleneck",
    "Complex System", "Network Theory", "Self-Organization",
    "Bifurcation Theory", "Chaos Theory",
})


# Pre-computed lowercase blocklist for case-insensitive matching.
# Data types, precision formats, and GPU identifiers should be blocked
# regardless of capitalization (float32, Float32, FLOAT32 are all noise).
_BLOCKLIST_LOWER: frozenset[str] = frozenset(b.lower() for b in _BLOCKLIST)


def _is_valid_entity(name: str) -> bool:
    """Return True only for names that could be genuine scientific concepts."""
    name = name.strip()

    # Length bounds
    if len(name) < 3 or len(name) > 80:
        return False

    # Must contain at least one ASCII letter
    if not re.search(r"[a-zA-Z]", name):
        return False

    # Reject names starting with a digit unless they use a dimensional prefix (2D, 3D, N6-)
    if name[0].isdigit() and not re.match(r'^\d+[A-Za-z]', name):
        return False

    # Explicit blocklist — case-insensitive so "Float32" == "float32" both blocked
    if name.lower() in _BLOCKLIST_LOWER:
        return False

    # Noise patterns
    if _NOISE_PATTERNS.search(name):
        return False

    # Reject truncated extractions — PDF line-break artifacts
    if name.endswith("-") or name.endswith("["):
        return False

    # Reject names with citation markers embedded: "Yang[1", "Smith[2,3]"
    if re.search(r"\[\d", name):
        return False

    # Reject pure lowercase single words (math variables: "f", "x", "alpha")
    if name.islower() and " " not in name and len(name) < 6:
        return False

    # Reject unbalanced parentheses (malformed extractions like "Mixtureof-Experts (MoE")
    if name.count("(") != name.count(")"):
        return False

    # Reject math/code noise: more than 35% non-alphanumeric chars (excluding safe punctuation)
    non_alpha = sum(1 for c in name if not c.isalnum() and c not in " -_.'()")
    if non_alpha > len(name) * 0.35:
        return False

    # Reject person names (various formats, no digits)
    if _PERSON_RE.match(name) and name not in _KNOWN_TWO_WORD_CONCEPTS:
        return False

    return True


# Module-level lazy singleton for the spaCy model
_NLP: Any | None = None


def _load_nlp() -> Any:
    import spacy

    for model in ("en_core_sci_sm", "en_core_web_sm"):
        try:
            nlp = spacy.load(model)
            logger.debug("Loaded spaCy model: {}", model)
            return nlp
        except OSError:
            continue

    raise RuntimeError(
        "No spaCy model found. Install one with:\n"
        "  python -m spacy download en_core_web_sm"
    )


def _get_nlp() -> Any:
    global _NLP
    if _NLP is None:
        _NLP = _load_nlp()
    return _NLP


def _build_phrase_matcher(nlp: Any) -> Any:
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for entity_type, phrases in _PHRASE_PATTERNS.items():
        patterns = [nlp.make_doc(p) for p in phrases]
        matcher.add(entity_type, patterns)
    return matcher


def _canonical_name(name: str) -> str:
    """Normalize surface form so case variants collapse to one canonical entry.

    Rules (applied in order):
    1. Strip leading "the " (case-insensitive)
    2. If all lowercase → title-case (e.g. "transformer" → "Transformer")
    3. Singularize the last word when safe — avoids duplicate concept notes
       for "MoE Transformers" vs "MoE Transformer" across all domains.
       Safe = last word ends in 's', len>3, not an all-caps acronym, not 'ss'.
    4. Otherwise keep original casing (acronyms like BLEU, GPT-3, mT5 stay as-is)
    """
    stripped = re.sub(r"^the\s+", "", name, flags=re.IGNORECASE).strip()
    if stripped == stripped.lower():
        stripped = stripped.title()

    # Singularize last word (applies after title-casing)
    words = stripped.split()
    if words:
        last = words[-1]
        if (last.endswith("s") and len(last) > 3
                and not last.isupper()
                and not last.endswith("ss")):
            words[-1] = last[:-1]
        stripped = " ".join(words)

    return stripped


# Entity type priority for deduplication: prefer specific types over generic.
# When the same concept appears as both "method" and "phenomenon" (common with
# NER noise), keep the more informative type across all future domains.
_TYPE_PRIORITY: dict[str, int] = {
    "method": 5, "material": 4, "chemical": 3, "gene": 2, "organism": 1, "phenomenon": 0
}


def _dedup_key(name: str) -> str:
    """Deduplication key that collapses hyphen variants and plural forms.

    "Feed-Forward Network", "Feed Forward Network", "Feed-Forward Networks"
    all map to the same key. Entity type is NOT part of the key — when the
    same concept appears as both "method" and "phenomenon", the higher-priority
    type wins (see _TYPE_PRIORITY). This prevents duplicate concept notes across
    all domains (bio, chem, physics, ML) where variant naming is common.
    """
    norm = name.lower().replace("-", " ").strip()
    # Collapse plural: strip trailing 's' only if safe (avoids "bus"→"bu")
    if norm.endswith("s") and len(norm) > 5 and not norm.endswith("ss"):
        norm = norm[:-1]
    return norm


_CITATION_RE = re.compile(
    r"^\s*[A-Z][^\.\?!]{0,120}\.\s*$"          # single sentence ending with period
)
_VERB_RE = re.compile(
    r"\b(is|are|was|were|can|enable|allow|use|provide|show|demonstrate|"
    r"perform|achieve|learn|train|predict|encode|represent|reduce|improve|"
    r"increase|decrease|apply|extend|propose|present|introduce)\b",
    re.IGNORECASE,
)


def _best_context(sentence: str) -> str:
    """Return *sentence* if it reads like a real descriptive sentence, else ''.

    Filters out citation titles and short captions that get picked up as
    context but don't actually define or describe the entity.

    A sentence is kept when it:
    - is at least 40 characters long
    - contains at least one verb from the scientific vocabulary above
    - does not look like a bare bibliography entry (Author, Year. Title.)
    """
    s = sentence.strip()
    if len(s) < 40:
        return ""
    if not _VERB_RE.search(s):
        return ""
    # Reject lines that are just a capitalised noun phrase ending with a period
    # and contain no comma or subordinate clause — typical of bibliography titles
    if re.match(r'^[A-Z][a-zA-Z\s\-]{5,80}\.$', s) and "," not in s:
        return ""
    return s


def extract_entities(doc: RawDocument, nlp: Any = None) -> list[ExtractedEntity]:
    """Extract scientific entities from *doc*.

    Pass *nlp* explicitly to inject a mock during tests; otherwise the shared
    singleton is used.
    """
    if nlp is None:
        nlp = _get_nlp()

    logger.info("Extracting entities from doc {!r}", doc.id)

    # Truncate to avoid hitting spaCy's max_length on very long papers
    text = doc.full_text[:100_000]
    spacy_doc = nlp(text)

    phrase_matcher = _build_phrase_matcher(nlp)
    entities: dict[tuple[str, str], ExtractedEntity] = {}

    # --- NER-based entities ---
    for ent in spacy_doc.ents:
        entity_type = _LABEL_MAP.get(ent.label_)
        if entity_type is None:
            continue
        name = _canonical_name(ent.text.strip())
        if not _is_valid_entity(name):
            continue
        # ORG-label entities are noisy — apply strict extra filtering.
        # This works across all domains (bio, chem, physics, ML) because NER's
        # ORG tag is a catch-all that picks up researcher names, institutions,
        # and paper-title fragments along with legitimate concept names.
        if ent.label_ == "ORG":
            words = name.split()
            has_digit = any(c.isdigit() for c in name)
            is_allcaps = name.replace("-", "").replace(" ", "").isupper()
            has_allcaps_word = any(w.isupper() and len(w) > 1 for w in words)

            # Single-word: must be all-caps acronym (BERT, WMT) or contain digit (T5, GPT-3)
            if len(words) == 1 and not (has_digit or is_allcaps):
                continue

            # 3+ word phrases: require digit or an all-caps word (acronym inside name).
            # "Examining Switch Transformer" → no → blocked
            # "Training Data and Batching" → no → blocked
            # "Google Research University..." → no → blocked
            # "GPT-3 Few-Shot Learning" → has digit → allowed
            if len(words) >= 3 and not has_digit and not has_allcaps_word:
                continue
        key = _dedup_key(name)
        existing = entities.get(key)
        if existing is None or _TYPE_PRIORITY.get(entity_type, 0) > _TYPE_PRIORITY.get(existing.entity_type, 0):
            entities[key] = ExtractedEntity(
                name=name,
                entity_type=entity_type,  # type: ignore[arg-type]
                source_doc_id=doc.id,
                context_sentence=_best_context(ent.sent.text.strip()),
            )

    # --- PhraseMatcher-based entities ---
    matches = phrase_matcher(spacy_doc)
    for match_id, start, end in matches:
        span = spacy_doc[start:end]
        name = _canonical_name(span.text.strip())
        entity_type = nlp.vocab.strings[match_id]
        if not _is_valid_entity(name):
            continue
        key = _dedup_key(name)
        existing = entities.get(key)
        if existing is None or _TYPE_PRIORITY.get(entity_type, 0) > _TYPE_PRIORITY.get(existing.entity_type, 0):
            entities[key] = ExtractedEntity(
                name=name,
                entity_type=entity_type,  # type: ignore[arg-type]
                source_doc_id=doc.id,
                context_sentence=_best_context(span.sent.text.strip()),
            )

    result = list(entities.values())
    logger.info("Found {} entities in doc {!r}", len(result), doc.id)
    return result
