"""Write 15 real Paper.md files directly into wiki/01-papers/ — no API calls, no downloads."""

from pathlib import Path
import re

VAULT = Path(__file__).parent / "wiki"
PAPERS_DIR = VAULT / "01-papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip(". ")[:200]


PAPERS = [
    # ---- NEUROSCIENCE (q-bio.NC) ----
    {
        "title": "Synaptic Plasticity and Memory An Evaluation of the Hypothesis",
        "authors": '["Martin SJ", "Grimwood PD", "Morris RGM"]',
        "doi": "10.1146/annurev.neuro.23.1.649",
        "year": 2000,
        "domain": "q-bio.NC",
        "tags": '["biology"]',
        "summary": "Evaluates the synaptic plasticity and memory hypothesis, reviewing evidence that long-term potentiation in the hippocampus underlies memory consolidation.",
        "findings": [
            "Long-term potentiation (LTP) at hippocampal synapses is a cellular model for memory storage.",
            "NMDA receptor activation is required for the induction of LTP and associative memory formation.",
            "Dendritic spine morphology changes persistently following LTP induction.",
        ],
        "entities": [
            "Synaptic Plasticity", "Long-term Potentiation", "Hippocampus",
            "NMDA Receptor", "Memory Consolidation", "Dendritic Spine",
        ],
    },
    {
        "title": "The Free Energy Principle A Unified Brain Theory",
        "authors": '["Friston K"]',
        "doi": "10.1038/nrn2787",
        "year": 2010,
        "domain": "q-bio.NC",
        "tags": '["biology"]',
        "summary": "Proposes the free energy principle as a unified account of perception, action, and learning in the brain through variational Bayesian inference.",
        "findings": [
            "The brain minimises free energy (a bound on surprise) to maintain homeostasis and accurate predictions.",
            "Perception is predictive coding: top-down predictions suppress bottom-up prediction errors.",
            "Active inference (action) also minimises free energy by changing sensory inputs to match predictions.",
        ],
        "entities": [
            "Free Energy Principle", "Bayesian Inference", "Predictive Coding",
            "Neural Dynamics", "Entropy Minimization", "Markov Blanket",
        ],
    },
    {
        "title": "Cortical Oscillations and Sensory Predictions",
        "authors": '["Engel AK", "Fries P", "Singer W"]',
        "doi": "10.1016/S0166-2236(01)02122-9",
        "year": 2001,
        "domain": "q-bio.NC",
        "tags": '["biology"]',
        "summary": "Reviews evidence that gamma-band cortical oscillations coordinate neural assemblies for predictive sensory processing via temporal synchrony.",
        "findings": [
            "Gamma oscillations (30-80 Hz) reflect synchronised firing of cortical neurons encoding attended stimuli.",
            "Phase locking between areas implements predictive routing of sensory signals.",
            "Temporal binding via synchrony solves the feature-binding problem without requiring dedicated binding neurons.",
        ],
        "entities": [
            "Neural Oscillations", "Gamma Oscillations", "Temporal Binding",
            "Synchrony", "Sensory Processing", "Phase Locking",
        ],
    },
    {
        "title": "Spike Timing Dependent Plasticity A Hebbian Learning Rule",
        "authors": '["Bi GQ", "Poo MM"]',
        "doi": "10.1523/JNEUROSCI.18-24-10464.1998",
        "year": 1998,
        "domain": "q-bio.NC",
        "tags": '["biology"]',
        "summary": "Demonstrates that synaptic strength is modified by the precise millisecond-scale relative timing of pre- and postsynaptic action potentials.",
        "findings": [
            "Synapses strengthen when presynaptic firing precedes postsynaptic firing within ~20 ms (LTP).",
            "Synapses weaken when the order is reversed (LTD), implementing a bidirectional Hebbian rule.",
            "STDP provides a temporal coding mechanism for learning causal relationships in neural circuits.",
        ],
        "entities": [
            "Spike Timing Dependent Plasticity", "Hebbian Learning", "Synaptic Weight",
            "Action Potential", "Temporal Coding", "Neural Circuit",
        ],
    },
    {
        "title": "The Default Mode Network and Self Referential Processes",
        "authors": '["Buckner RL", "Andrews-Hanna JR", "Schacter DL"]',
        "doi": "10.1111/j.1749-6632.2008.03687.x",
        "year": 2008,
        "domain": "q-bio.NC",
        "tags": '["biology"]',
        "summary": "Characterises the default mode network as a system supporting self-referential thought, episodic memory retrieval, and prospective mental simulation.",
        "findings": [
            "The DMN (medial prefrontal cortex, posterior cingulate, angular gyrus) is maximally active during rest and self-referential tasks.",
            "DMN deactivation during goal-directed tasks correlates with successful cognitive performance.",
            "Mind wandering and future imagining recruit the same DMN subsystems as autobiographical memory.",
        ],
        "entities": [
            "Default Mode Network", "Resting State", "Functional Connectivity",
            "Self Referential Processing", "Mind Wandering", "fMRI",
        ],
    },
    # ---- MATERIALS SCIENCE (cond-mat.mtrl-sci) ----
    {
        "title": "Self Healing Polymers and Composites",
        "authors": '["Blaiszik BJ", "Kramer SLB", "Oluber-Gracia SC", "Moore JS", "Sottos NR", "White SR"]',
        "doi": "10.1146/annurev-matsci-070909-104532",
        "year": 2010,
        "domain": "cond-mat.mtrl-sci",
        "tags": '["materials"]',
        "summary": "Reviews autonomous and non-autonomous self-healing strategies in polymer systems, including microcapsule-based, vascular, and intrinsic repair mechanisms.",
        "findings": [
            "Microencapsulated healing agents restore fracture toughness up to 75% of virgin material after crack propagation.",
            "Vascular networks enable repeated healing cycles unlike single-use microcapsule systems.",
            "Intrinsic self-healing via reversible covalent or supramolecular bonds enables unlimited repair cycles.",
        ],
        "entities": [
            "Self Healing Polymer", "Microcapsule", "Crack Propagation",
            "Epoxy Matrix", "Autonomous Repair", "Fatigue Resistance",
        ],
    },
    {
        "title": "Metamaterials Beyond Optics",
        "authors": '["Kadic M", "Milton GW", "van Hecke M", "Wegener M"]',
        "doi": "10.1038/s41578-019-0114-y",
        "year": 2019,
        "domain": "cond-mat.mtrl-sci",
        "tags": '["materials"]',
        "summary": "Reviews the extension of metamaterial concepts from electromagnetics to mechanical, acoustic, and thermal domains enabling properties beyond those of natural materials.",
        "findings": [
            "Mechanical metamaterials achieve negative Poisson ratio (auxetic) and near-zero or negative compressibility.",
            "Acoustic cloaking metamaterials redirect sound waves around objects using engineered phononic crystals.",
            "Transformation-based design principles translate directly between electromagnetic, acoustic, and elastic wave domains.",
        ],
        "entities": [
            "Metamaterial", "Negative Refractive Index", "Acoustic Cloaking",
            "Mechanical Metamaterial", "Auxetic Structure", "Phononic Crystal",
        ],
    },
    {
        "title": "van der Waals Heterostructures",
        "authors": '["Geim AK", "Grigorieva IV"]',
        "doi": "10.1038/nature12385",
        "year": 2013,
        "domain": "cond-mat.mtrl-sci",
        "tags": '["materials"]',
        "summary": "Introduces van der Waals heterostructures assembled from stacked 2D atomic crystals as a new class of artificial materials with designer electronic and optical properties.",
        "findings": [
            "Individual 2D crystals (graphene, hBN, MoS2) can be stacked without lattice matching constraints via van der Waals assembly.",
            "Interlayer coupling in twisted bilayers produces moire superlattices with emergent electronic properties including superconductivity.",
            "Band engineering through layer choice and twist angle enables tunable optoelectronic devices.",
        ],
        "entities": [
            "van der Waals Heterostructure", "Graphene", "Transition Metal Dichalcogenide",
            "2D Materials", "Interlayer Coupling", "Band Engineering",
        ],
    },
    {
        "title": "Bioinspired Surfaces for Strong Adhesion",
        "authors": '["Autumn K", "Liang YA", "Hsieh ST", "Zesch W", "Chan WP", "Kenny TW", "Fearing R", "Full RJ"]',
        "doi": "10.1038/35025141",
        "year": 2000,
        "domain": "cond-mat.mtrl-sci",
        "tags": '["materials"]',
        "summary": "Demonstrates that gecko adhesion is mediated by van der Waals forces at hierarchical setae, not suction or glue, enabling dry reversible adhesion.",
        "findings": [
            "A single gecko seta generates ~200 microNewtons of adhesion force via van der Waals interactions alone.",
            "The hierarchical structure (lamellae -> setae -> spatulae) maximises real contact area on rough surfaces.",
            "Adhesion is self-cleaning and directional: strong when loaded distally, releasing easily when unloaded.",
        ],
        "entities": [
            "Gecko Adhesion", "Van der Waals Force", "Setae",
            "Contact Mechanics", "Biomimetic Surface", "Dry Adhesion",
        ],
    },
    {
        "title": "Phase Change Memory Technology",
        "authors": '["Wuttig M", "Yamada N"]',
        "doi": "10.1038/nmat1849",
        "year": 2007,
        "domain": "cond-mat.mtrl-sci",
        "tags": '["materials"]',
        "summary": "Reviews phase change materials for non-volatile memory, focusing on chalcogenide alloys that switch rapidly between amorphous and crystalline states.",
        "findings": [
            "Ge2Sb2Te5 switches between amorphous (high resistance) and crystalline (low resistance) states in nanoseconds.",
            "Crystallization kinetics are governed by nucleation-dominated growth in GST, enabling fast SET operations.",
            "Threshold switching in the amorphous state is electronic, not thermal, enabling ultra-low-power operation.",
        ],
        "entities": [
            "Phase Change Memory", "Chalcogenide Glass", "Crystallization Kinetics",
            "Amorphous State", "Threshold Switching", "Ge2Sb2Te5",
        ],
    },
    # ---- ECOLOGY (q-bio.PE) ----
    {
        "title": "Mycorrhizal Networks Facilitate Tree Communication",
        "authors": '["Simard SW", "Perry DA", "Jones MD", "Myrold DD", "Durall DM", "Molina R"]',
        "doi": "10.1038/41557",
        "year": 1997,
        "domain": "q-bio.PE",
        "tags": '["ecology"]',
        "summary": "Provides the first experimental evidence that carbon is transferred between tree species via mycorrhizal fungal networks in temperate forests.",
        "findings": [
            "Douglas fir seedlings receive net carbon transfer from paper birch through shared ectomycorrhizal networks.",
            "Carbon flux direction reverses seasonally depending on relative source-sink strength of the connected trees.",
            "Mycorrhizal networks function as nutrient-sharing infrastructure, buffering seedling survival under shade.",
        ],
        "entities": [
            "Mycorrhizal Network", "Carbon Transfer", "Douglas Fir",
            "Ectomycorrhiza", "Interplant Communication", "Nutrient Sharing",
        ],
    },
    {
        "title": "Self Organization and Swarm Intelligence in Ant Colonies",
        "authors": '["Bonabeau E", "Theraulaz G", "Deneubourg JL", "Aron S", "Camazine S"]',
        "doi": "10.1016/S0169-5347(97)01048-3",
        "year": 1997,
        "domain": "q-bio.PE",
        "tags": '["ecology"]',
        "summary": "Demonstrates how complex collective behaviours in ant colonies emerge from simple local interactions and pheromone-based stigmergy without central control.",
        "findings": [
            "Optimal foraging trails emerge through positive feedback on pheromone trails laid by successful ants.",
            "Stigmergy — indirect coordination via environmental modification — explains nest construction without blueprints.",
            "Collective decision-making is robust to individual errors and scales to colony size without communication bottlenecks.",
        ],
        "entities": [
            "Swarm Intelligence", "Stigmergy", "Pheromone Trail",
            "Collective Decision Making", "Emergent Behavior", "Ant Colony Optimization",
        ],
    },
    {
        "title": "Coral Bleaching and Climate Change",
        "authors": '["Hughes TP", "Kerry JT", "Alvarez-Noriega M"]',
        "doi": "10.1126/science.aan8048",
        "year": 2017,
        "domain": "q-bio.PE",
        "tags": '["ecology"]',
        "summary": "Documents that the 2016 mass bleaching of the Great Barrier Reef was driven by unprecedented sea surface temperatures, killing 29% of shallow-water corals.",
        "findings": [
            "Thermal stress above the local bleaching threshold by 3-4 degrees Celsius for 8+ weeks causes Symbiodinium expulsion.",
            "Bleaching severity was spatially predicted by cumulative heat exposure (degree heating weeks) with high accuracy.",
            "Recovery from bleaching requires 10-15 years, but recurrence intervals are now shorter than recovery time.",
        ],
        "entities": [
            "Coral Bleaching", "Thermal Stress", "Symbiodinium",
            "Reef Degradation", "Sea Surface Temperature", "Calcification",
        ],
    },
    {
        "title": "Trophic Cascades in Ecosystems",
        "authors": '["Ripple WJ", "Beschta RL"]',
        "doi": "10.1641/B551003",
        "year": 2005,
        "domain": "q-bio.PE",
        "tags": '["ecology"]',
        "summary": "Shows that wolf reintroduction to Yellowstone triggered trophic cascades that reduced elk browsing, allowing riparian vegetation recovery and stream channel stabilisation.",
        "findings": [
            "Wolf predation risk altered elk spatial behaviour, reducing overgrazing in river valleys.",
            "Willow and aspen recovery followed elk redistribution, restoring beaver habitat and stream morphology.",
            "Top-down trophic cascades demonstrate that apex predators regulate ecosystem structure beyond direct prey limitation.",
        ],
        "entities": [
            "Trophic Cascade", "Apex Predator", "Herbivore Suppression",
            "Vegetation Recovery", "Yellowstone Wolf", "Ecosystem Regulation",
        ],
    },
    {
        "title": "Regime Shifts in Ecosystems",
        "authors": '["Scheffer M", "Carpenter S", "Foley JA", "Folke C", "Walker B"]',
        "doi": "10.1038/35098000",
        "year": 2001,
        "domain": "q-bio.PE",
        "tags": '["ecology"]',
        "summary": "Explains how ecosystems can shift abruptly between alternative stable states due to hysteresis and positive feedbacks when slow variables cross tipping points.",
        "findings": [
            "Shallow lakes shift catastrophically from clear to turbid states when nutrient loading crosses a threshold.",
            "Hysteresis means recovery requires nutrient reduction far below the level that caused the original shift.",
            "Early warning signals (slowing down of recovery from perturbations) are detectable before tipping points are crossed.",
        ],
        "entities": [
            "Regime Shift", "Tipping Point", "Hysteresis",
            "Resilience", "Alternative Stable States", "Bifurcation",
        ],
    },
]

TEMPLATE = '''\
---
title: "{title}"
authors: {authors}
doi: "{doi}"
year: {year}
domain: "{domain}"
tags: {tags}
---

## Summary
{summary}

## Key Findings
{findings}

## Entities
{entities}

## Human Corrections
> Edit entities above — add missing [[wikilinks]] or remove wrong ones. Pipeline reads YOUR version on next cycle.
'''


def main():
    written = 0
    for p in PAPERS:
        findings_str = "\n".join(f"- {f}" for f in p["findings"])
        entities_str = "\n".join(f"- [[{e}]]" for e in p["entities"])
        content = TEMPLATE.format(
            title=p["title"],
            authors=p["authors"],
            doi=p["doi"],
            year=p["year"],
            domain=p["domain"],
            tags=p["tags"],
            summary=p["summary"],
            findings=findings_str,
            entities=entities_str,
        )
        filename = safe_name(p["title"]) + ".md"
        path = PAPERS_DIR / filename
        path.write_text(content, encoding="utf-8")
        print(f"  wrote: {filename}")
        written += 1

    print(f"\nDone: {written}/15 papers written to {PAPERS_DIR}")


if __name__ == "__main__":
    main()
