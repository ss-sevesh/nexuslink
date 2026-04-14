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
    r"|^[=|`>\[<~\u201c\u2018\"\']"        # starts with markup or quote chars
    r"|\|\|"                                # double pipe
    r"|et al"                               # citation fragments
    r"|~~"                                  # strikethrough markup
    r"|^[Bb][Rr]>"                          # HTML artifact (Br>, br>)
    r"|^[^\x00-\x7F]"                       # starts with non-ASCII (↑, ×, etc.)
    r"|[×÷±∑∏∂∫√|\u2212\u00b1]"            # math operators, pipe, unicode minus/plus
    r"|^(Conference|Proceedings|Journal|Workshop|Symposium|Annual|International)\s"
                                            # academic venue names
    r"|\b(University|Institute|Laboratory|College|Church|Brain Team|Research Lab)\b"
                                            # institutional names
    r"|\b(Conference on|Processing Systems|Technology Conference)\b"
                                            # venue/journal name fragments
    r"|,\s*(Inc\.|Ltd\.|Corp\.|LLC)"        # company suffixes
    r"|\s&\s[A-Z][a-z]+"                   # "X & Lastname" citation patterns
    r"|^[A-Z]{2,5}\d{2,4}$"               # citation codes like RNSS18, ACL19
    r"|\.\w{2,4}$"                          # file extensions / URLs (.com, .org)
    r"|^[A-Z]\.\s"                          # initial + period (e.g. "D. Switch")
    r"|\[\]"                                # square brackets (table/code artifacts)
    r"|\+\s"                                # arithmetic + operator
    r"|,\s*[A-Z]"                           # comma before capital (list fragments)
    r"|'\d{2}\b"                            # apostrophe+year: WMT'16
    r"|\bto\s+[A-Z][a-z]"                  # "to Tensorflow" — description fragment
    r"|\w*[^\x00-\x7F]\w*\s+[A-Z][a-z]+"  # non-ASCII word followed by surname
)

# Person-name patterns (various formats that appear in paper author/citation sections)
_PERSON_RE = re.compile(
    r"^[A-Z][a-z]+\s+[A-Z][a-z]+$"              # "John Smith"
    r"|^[A-Z][a-z]+\s+[A-Z]+\s+[A-Z][a-z]+$"    # "Quoc VV Le", "J. R. Jones"
    r"|^[A-Z]\.\s+[A-Z][a-z]+"                   # "J. Smith" (with period)
)

# Two-word technical terms that superficially look like person names
_KNOWN_TWO_WORD_CONCEPTS: frozenset[str] = frozenset({
    "Deep Learning", "Machine Learning", "Transfer Learning",
    "Switch Transformer", "Mesh Tensorflow", "Natural Language",
    "Neural Network", "Language Model", "Sparse Transformer",
    "Few-Shot", "Zero-Shot", "Common Crawl", "Winograd Schema",
    "Byte Pair", "Attention Mechanism", "Gradient Descent",
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

    # Reject names starting with a digit (metric values, table numbers, etc.)
    if name[0].isdigit():
        return False

    # Explicit blocklist — case-insensitive so "Float32" == "float32" both blocked
    if name.lower() in _BLOCKLIST_LOWER:
        return False

    # Noise patterns
    if _NOISE_PATTERNS.search(name):
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
                context_sentence=ent.sent.text.strip(),
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
                context_sentence=span.sent.text.strip(),
            )

    result = list(entities.values())
    logger.info("Found {} entities in doc {!r}", len(result), doc.id)
    return result
