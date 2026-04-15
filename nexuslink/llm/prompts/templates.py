"""Jinja2 prompt templates for each LLM pipeline stage.

Each template has a *system* component (sets the AI's role) and a *user*
component (the per-call variable payload).  Use ``render_template`` to render
the user portion and ``get_system_prompt`` to retrieve the system string.
"""

from __future__ import annotations

from jinja2 import BaseLoader, Environment

_env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)


# ===========================================================================
# System prompts (static — one per stage)
# ===========================================================================

_SYS_GENERATION = (
    "You are a cross-domain research scientist. Your job is to identify non-obvious "
    "connections between scientific fields and generate novel, testable hypotheses. "
    "Think like a scientist who has read deeply across physics, biology, chemistry, "
    "materials science, computer science, and medicine — and who can see structural "
    "similarities that domain specialists miss."
)

_SYS_CRITIQUE = (
    "You are a rigorous peer reviewer evaluating cross-domain hypotheses. "
    "Apply the standards of top-tier journals: specificity, falsifiability, and "
    "evidence quality. Be tough but fair — identify genuine weaknesses, not pedantic ones."
)

_SYS_REFINEMENT = (
    "You are a senior research scientist refining hypotheses based on expert peer review. "
    "Preserve the core cross-domain insight while addressing all identified weaknesses. "
    "Make predictions more quantitative and experiments more specific wherever possible."
)

_SYS_REPORT = (
    "You are a scientific report writer creating a cross-domain hypothesis report for "
    "funding agencies and research teams. Your audience includes expert reviewers from "
    "all domains listed. Write clearly, precisely, and with appropriate scientific rigour. "
    "Use [[wikilinks]] for every paper title and concept name so the report integrates "
    "with the Obsidian research vault."
)


# ===========================================================================
# User templates (Jinja2, rendered per call)
# ===========================================================================

_USER_GENERATION_SRC = """\
## Output Structure (example only — do NOT reproduce this content)

The JSON below shows the required fields and format. The hypothesis content is fictional.
Generate your own hypotheses solely from the bridges listed further below.

```json
[
  {
    "statement": "If [specific quantitative finding from domain A] and [named mechanism from domain B], then [novel, measurable prediction distinct from either domain alone].",
    "domains_spanned": ["domain_a", "domain_b"],
    "suggested_experiments": [
      "Specific experiment with named technique and measurable outcome.",
      "Second experiment with protocol and expected result.",
      "Third experiment targeting the falsifiable prediction."
    ],
    "bridge_index": 0,
    "confidence": 0.75,
    "reasoning": "Why the two domains connect mechanistically."
  }
]
```

`bridge_index` is the 0-based index of the bridge in the list below that inspired this hypothesis.

---

## Cross-Domain Bridges to Analyse

{% for bridge in bridges %}
**Bridge {{ loop.index }}:**
  {{ bridge.entity_a }} ({{ bridge.domain_a }}, {{ bridge.entity_type_a }})
    ←[{{ bridge.bridge_type }}, similarity={{ "%.3f"|format(bridge.similarity_score) }}]→
  {{ bridge.entity_b }} ({{ bridge.domain_b }}, {{ bridge.entity_type_b }})
{% endfor %}

## Knowledge Graph Context

Domains represented: {{ domains | join(", ") }}
Total concepts in graph: {{ total_concepts }}

## Requirements

For each bridge (or combination of bridges), generate ONE novel hypothesis:
- Structure: "If [specific finding from domain A] and [mechanism from domain B], then [novel, testable prediction]"
- Must be *falsifiable* — state conditions under which it would be disproved
- Predictions must be quantitative where possible (mention measurable quantities)
- Experiments must be realistic with current or near-future technology (name the technique)
- Span must be genuinely cross-domain — experts in a single field alone would not generate it

## Output Format

Respond with a JSON array only — no markdown, no explanation outside the JSON:

[
  {
    "statement": "If [finding A from domain X] and [mechanism B from domain Y], then [novel prediction]",
    "domains_spanned": ["domain1", "domain2"],
    "suggested_experiments": [
      "Specific experiment 1 with named technique and measurable outcome",
      "Specific experiment 2 with named technique and measurable outcome",
      "Specific experiment 3 with named technique and measurable outcome"
    ],
    "confidence": 0.75,
    "reasoning": "Explanation of the cross-domain logic — why domains A and B connect here"
  }
]
"""

_USER_CRITIQUE_SRC = """\
## Hypothesis Under Review

**Statement:** {{ hypothesis.statement }}

**Domains spanned:** {{ hypothesis.domains_spanned | join(", ") }}

**Suggested experiments:**
{% for exp in hypothesis.suggested_experiments %}
{{ loop.index }}. {{ exp }}
{% endfor %}

**Reasoning:** {{ hypothesis.raw_reasoning }}

## Scoring Rubric

Rate each dimension 1–10 using these anchors:

**Novelty (N):**
  1 = already well-known; 5 = somewhat surprising; 10 = genuinely unexpected even to experts

**Feasibility (F):**
  1 = requires impossible technology; 5 = doable within 10 years; 10 = executable today

**Impact (I):**
  1 = minor result; 5 = significant field advance; 10 = paradigm-shifting if confirmed

**Mechanistic Depth (M):**
  1 = vague analogy only; 5 = plausible mechanism described; 10 = precise molecular/physical mechanism with known intermediates

**Falsifiability (Fs):**
  1 = untestable or unfalsifiable; 5 = testable with significant effort; 10 = specific quantitative prediction with clear null hypothesis

## Output Format

Respond with JSON only — no preamble, no explanation outside the JSON.
All scores are integers 1–10. Use the full range — do not anchor to example values.

{
  "novelty_score": <int 1-10>,
  "feasibility_score": <int 1-10>,
  "impact_score": <int 1-10>,
  "mechanistic_depth": <int 1-10>,
  "falsifiability_score": <int 1-10>,
  "strengths": [
    "Strength 1",
    "Strength 2"
  ],
  "weaknesses": [
    "Specific weakness or gap 1",
    "Specific weakness or gap 2"
  ],
  "missing_evidence": [
    "Evidence that would significantly strengthen this hypothesis"
  ],
  "verdict": "promising | speculative | weak",
  "critique_summary": "2–3 sentence assessment of the hypothesis's scientific merit and most promising direction."
}
"""

_USER_REFINEMENT_SRC = """\
## Original Hypothesis

**Statement:** {{ hypothesis.statement }}

**Domains spanned:** {{ hypothesis.domains_spanned | join(", ") }}

## Peer Review Scores

- Novelty:     {{ scored.novelty_score }}/10
- Feasibility: {{ scored.feasibility_score }}/10
- Impact:      {{ scored.impact_score }}/10

**Critique summary:** {{ scored.critique_summary }}

**Weaknesses to address:**
{% for w in scored.weaknesses %}
- {{ w }}
{% endfor %}

{% if scored.weaknesses is defined and scored.weaknesses | length == 0 %}
(No specific weaknesses identified — refine for precision only.)
{% endif %}

## Original Experiments

{% for exp in hypothesis.suggested_experiments %}
{{ loop.index }}. {{ exp }}
{% endfor %}

## Revision Instructions

1. Keep the "If [A] and [B], then [C]" structure
2. Make predictions more specific and quantitative (add numbers, units, thresholds)
3. Replace vague experiments with named protocols, assays, or instruments
4. Address each listed weakness explicitly in either the statement or experiments
5. Increase specificity of domain references where possible

## Output Format

Respond with JSON only:

{
  "revised_statement": "If [...] and [...], then [...]",
  "addressed_weaknesses": [
    "Weakness 1: addressed by [specific change made]"
  ],
  "revised_experiments": [
    "Revised experiment 1 with specific protocol and measurable outcome",
    "Revised experiment 2 with specific protocol and measurable outcome",
    "Revised experiment 3 with specific protocol and measurable outcome"
  ],
  "revised_confidence": 0.82
}
"""

_USER_REPORT_SRC = """\
## Top Hypotheses (Ranked by Composite Score)

{% for h in hypotheses %}
### Hypothesis {{ loop.index }} — Score: {{ "%.1f"|format(h.composite_score) }}/10

**Statement:** {{ h.statement }}
**Domains:** {{ h.domains_spanned | join(" × ") }}
**Scores:** N={{ h.novelty_score }} / F={{ h.feasibility_score }} / I={{ h.impact_score }}
**Top experiment:** {{ h.suggested_experiments[0] if h.suggested_experiments else "N/A" }}
{% if h.weaknesses %}
**Known weaknesses:** {{ h.weaknesses | join("; ") }}
{% endif %}
**Evidence bridges:** {{ h.evidence_bridges | join(", ") }}

{% endfor %}

## Knowledge Graph Statistics

- Papers analysed: {{ stats.papers_processed }}
- Concepts extracted: {{ stats.total_concepts }}
- Cross-domain bridges detected: {{ stats.total_bridges }}
- Domains covered: {{ stats.domains | join(", ") }}

## Tasks

Write the following two sections for the research report.
Use [[wikilinks]] for every paper title and concept name (e.g. [[Casimir effect]], [[gecko adhesion]]).

1. **Executive Summary** (350–500 words): Explain the cross-domain analysis methodology,
   highlight the most significant hypotheses, and identify the most promising research
   directions. Mention domain names with [[wikilinks]].

2. **Cross-Domain Analysis Narrative** (200–300 words): Describe the conceptual
   landscape — which domains connect most strongly, why, and what that implies about
   underlying structural similarities. Use [[wikilinks]] for key concepts.

## Output Format

Respond with JSON only:

{
  "executive_summary": "...",
  "cross_domain_narrative": "..."
}
"""


# ===========================================================================
# Template registry and rendering API
# ===========================================================================

_REGISTRY: dict[str, tuple[str, str]] = {
    "hypothesis_generation": (_SYS_GENERATION, _USER_GENERATION_SRC),
    "hypothesis_critique": (_SYS_CRITIQUE, _USER_CRITIQUE_SRC),
    "hypothesis_refinement": (_SYS_REFINEMENT, _USER_REFINEMENT_SRC),
    "report_synthesis": (_SYS_REPORT, _USER_REPORT_SRC),
}

# Pre-compiled template objects (avoids re-parsing on every call)
_COMPILED: dict[str, object] = {
    name: _env.from_string(user_src)
    for name, (_, user_src) in _REGISTRY.items()
}


def render_template(template_name: str, **kwargs) -> str:
    """Render the user portion of a named Jinja2 template.

    Parameters
    ----------
    template_name:
        One of: ``hypothesis_generation``, ``hypothesis_critique``,
        ``hypothesis_refinement``, ``report_synthesis``.
    **kwargs:
        Variables forwarded to the Jinja2 template.

    Raises
    ------
    KeyError
        If *template_name* is not in the registry.
    """
    if template_name not in _COMPILED:
        raise KeyError(
            f"Unknown template {template_name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _COMPILED[template_name].render(**kwargs)  # type: ignore[union-attr]


def get_system_prompt(template_name: str) -> str:
    """Return the system prompt string for *template_name*."""
    if template_name not in _REGISTRY:
        raise KeyError(f"Unknown template: {template_name!r}")
    return _REGISTRY[template_name][0]


# ---------------------------------------------------------------------------
# Convenience wrappers (preserve backward-compat with existing callers)
# ---------------------------------------------------------------------------

def render_hypothesis_generation(
    bridges: list,
    domains: list[str],
    total_concepts: int,
) -> str:
    return render_template(
        "hypothesis_generation",
        bridges=bridges,
        domains=domains,
        total_concepts=total_concepts,
    )


def render_hypothesis_critique(hypothesis: object) -> str:
    return render_template("hypothesis_critique", hypothesis=hypothesis)


def render_hypothesis_refinement(hypothesis: object, scored: object) -> str:
    return render_template("hypothesis_refinement", hypothesis=hypothesis, scored=scored)


def render_report_synthesis(hypotheses: list, stats: dict) -> str:
    return render_template("report_synthesis", hypotheses=hypotheses, stats=stats)
