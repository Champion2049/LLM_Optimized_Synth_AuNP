"""
Deterministic synthesis-optimization recommender for AuNP synthesis.

This module is the source of truth for synthesis-optimization advice. It maps each
out-of-range property to a chemist-written recommendation via a deterministic lookup:

    (Synthesis Method, Property, Direction of deviation) -> pre-validated recommendation

Every cell in the table is written and approved in advance by a domain chemist, so the
output carries zero hallucination risk and a one-line mechanistic justification a reviewer
can check against the literature.

Pipeline
--------
1.  ``infer_synthesis_method``   - derive the method from the reducer/stabilizer/morphology
                                   the user actually selected.
2.  ``compute_deviations``       - the Delta_k deviation score (Eq. 16): which property is
                                   out of range, in which direction, and by how much.
3.  ``lookup_recommendation``    - the deterministic (method, property, direction) lookup.
4.  ``build_recommendation_report`` - rank deviations by |Delta_k|, concatenate the top
                                   templates, and append a synergy/conflict note.

The only "reasoning" left is the synergy note in step 4 (do two fixes reinforce or
conflict?), and even that is computed deterministically from set logic on the reagents each
recommendation adjusts. ``generate_explanation`` is an optional layer that expands the
pre-approved recommendations into a fuller, teaching-quality explanation; it is tightly
grounded in this table and never decides or changes the chemistry.
"""

from typing import Dict, List, Optional, Tuple, Any

# --- Canonical synthesis-method names ---------------------------------------
TURKEVICH = "Turkevich"
SEED_MEDIATED = "Seed-Mediated"
BRUST = "Brust-Schiffrin"
GREEN = "Polyol/Green"

# --- Canonical deviation directions -----------------------------------------
TOO_HIGH = "too_high"
TOO_LOW = "too_low"

# Some scripts label the same property differently (e.g. a trailing "_%"). Map those
# variants onto the canonical rule-table keys so a single rule table serves every pipeline.
PROPERTY_ALIASES = {
    "Drug_Loading_Efficiency_%": "Drug_Loading_Efficiency",
    "Targeting_Efficiency_%": "Targeting_Efficiency",
    "Cytotoxicity_%": "Cytotoxicity",
}


def canonical_property(prop: str) -> str:
    """Resolve a property label to its canonical rule-table key."""
    return PROPERTY_ALIASES.get(prop, prop)


# --- Display helpers ---------------------------------------------------------
PROPERTY_UNITS = {
    "Particle_Size_nm": "nm",
    "Particle_Width_nm": "nm",
    "Aspect_Ratio": "",
    "PDI": "",
    "Zeta_Potential_mV": "mV",
    "Drug_Loading_Efficiency": "%",
    "Targeting_Efficiency": "%",
    "Cytotoxicity": "%",
}

# Human-readable label for each reagent/parameter a recommendation adjusts.
# Used only to phrase the synergy/conflict note.
REAGENT_LABELS = {
    "citrate": "citrate",
    "temperature": "reaction temperature",
    "pH": "pH",
    "thiol": "thiol capping ligand",
    "NaBH4": "NaBH4 reductant",
    "AgNO3": "AgNO3 (silver) additive",
    "seed": "seed quantity",
    "growth_volume": "growth-solution volume",
    "gold": "gold precursor",
    "extract": "plant-extract bioreductant",
    "CTAB": "residual CTAB",
    "PEG": "PEG-thiol coating",
    "ligand": "surface-ligand functionalization",
    "targeting_ligand": "targeting-ligand density",
    "wash": "purification / wash step",
    "drug": "drug payload",
}

_DIRECTION_GERUND = {
    "increase": "increasing",
    "decrease": "decreasing",
    "add": "adding",
    "modify": "modifying",
    "standardize": "standardizing",
}


# ===========================================================================
# 1. THE RULE TABLE  --  (method, property, direction) -> recommendation
# ===========================================================================
# Each leaf is a dict with:
#   recommendation : str  - the chemist-written action
#   mechanism      : str  - one-line "why this works"
#   adjusts        : list of (reagent_token, direction) tuples, where direction is one of
#                    increase / decrease / add / modify / standardize. Used to detect when
#                    two recommendations reinforce or conflict.
#
# Cells are intentionally not exhaustive: when a (method, property, direction) combination
# has no pre-validated rule, the engine reports "flagged for chemist review" rather than
# inventing chemistry. This is the deliberate "always correct, sometimes silent" behaviour.

SYNTHESIS_METHOD_RULES: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    TURKEVICH: {
        "Particle_Size_nm": {
            TOO_HIGH: {
                "recommendation": "Increase the citrate-to-gold ratio.",
                "mechanism": "More citrate ions create more nucleation sites, so each "
                             "nucleus receives less gold and the final particles are smaller.",
                "adjusts": [("citrate", "increase")],
            },
            TOO_LOW: {
                "recommendation": "Decrease the citrate-to-gold ratio, or lower the reaction temperature.",
                "mechanism": "Fewer nuclei means more gold is deposited per particle, "
                             "increasing the final size.",
                "adjusts": [("citrate", "decrease"), ("temperature", "decrease")],
            },
        },
        "Particle_Width_nm": {
            TOO_HIGH: {
                "recommendation": "Increase the citrate-to-gold ratio.",
                "mechanism": "Turkevich particles are near-spherical, so width tracks "
                             "diameter; more citrate gives more, smaller nuclei.",
                "adjusts": [("citrate", "increase")],
            },
            TOO_LOW: {
                "recommendation": "Decrease the citrate-to-gold ratio, or lower the reaction temperature.",
                "mechanism": "Fewer nuclei deposit more gold per particle, increasing width.",
                "adjusts": [("citrate", "decrease"), ("temperature", "decrease")],
            },
        },
        "PDI": {
            TOO_HIGH: {
                "recommendation": "Raise the reaction temperature for faster, more simultaneous nucleation.",
                "mechanism": "Rapid nucleation shortens the time window over which particles "
                             "of different ages grow, narrowing the size distribution.",
                "adjusts": [("temperature", "increase")],
            },
        },
        "Zeta_Potential_mV": {
            TOO_HIGH: {  # "not negative enough"
                "recommendation": "Increase citrate concentration or raise the pH slightly.",
                "mechanism": "More adsorbed citrate anions (and greater deprotonation at "
                             "higher pH) strengthen the negative surface charge.",
                "adjusts": [("citrate", "increase"), ("pH", "increase")],
            },
        },
        "Drug_Loading_Efficiency": {
            TOO_LOW: {
                "recommendation": "Add a post-synthesis thiol-PEG functionalization step to introduce drug-binding groups.",
                "mechanism": "The citrate monolayer is weakly bound and offers few conjugation "
                             "handles; thiol anchoring adds functional surface density for drug loading.",
                "adjusts": [("ligand", "add")],
            },
        },
        "Targeting_Efficiency": {
            TOO_LOW: {
                "recommendation": "Conjugate a targeting ligand (folate or antibody) to the citrate surface via EDC/NHS coupling.",
                "mechanism": "Bare citrate-capped particles carry no receptor-binding ligand; "
                             "adding one enables selective cancer-cell uptake.",
                "adjusts": [("targeting_ligand", "add")],
            },
        },
        "Cytotoxicity": {
            TOO_HIGH: {
                "recommendation": "Add a purification/wash step to remove residual gold precursor and excess free citrate.",
                "mechanism": "Unreacted HAuCl4 and excess reagents are a known off-target "
                             "toxicity source, separate from the AuNP core itself.",
                "adjusts": [("wash", "add")],
            },
            TOO_LOW: {
                "recommendation": "Increase the conjugated drug payload, or reduce shielding-ligand density at the target.",
                "mechanism": "Low cytotoxicity to cancer cells usually reflects too little "
                             "delivered drug or over-shielding of the active surface.",
                "adjusts": [("drug", "increase")],
            },
        },
    },

    SEED_MEDIATED: {
        "Aspect_Ratio": {
            TOO_LOW: {
                "recommendation": "Increase the AgNO3 additive concentration relative to seed.",
                "mechanism": "Silver underpotential deposition is the classic shape-directing "
                             "mechanism that promotes anisotropic, rod-like growth.",
                "adjusts": [("AgNO3", "increase")],
            },
            TOO_HIGH: {
                "recommendation": "Decrease the AgNO3 additive concentration, or increase seed concentration.",
                "mechanism": "Less silver-directed anisotropy and more seeds give lower-aspect, "
                             "more isotropic particles.",
                "adjusts": [("AgNO3", "decrease"), ("seed", "increase")],
            },
        },
        "Particle_Width_nm": {
            TOO_HIGH: {  # rods too thick -> aspect ratio too low
                "recommendation": "Increase the AgNO3 additive concentration relative to seed.",
                "mechanism": "Silver-directed anisotropic growth narrows rods, raising the "
                             "length-to-width aspect ratio.",
                "adjusts": [("AgNO3", "increase")],
            },
        },
        "Particle_Size_nm": {
            TOO_HIGH: {
                "recommendation": "Use smaller (or more) seed particles, or reduce the growth-solution volume.",
                "mechanism": "Less gold is available per seed during overgrowth, yielding "
                             "smaller final particles.",
                "adjusts": [("seed", "increase"), ("growth_volume", "decrease")],
            },
            TOO_LOW: {
                "recommendation": "Use larger or fewer seeds, or increase the growth-solution gold concentration.",
                "mechanism": "More gold deposited per seed during overgrowth increases the final size.",
                "adjusts": [("seed", "decrease"), ("gold", "increase")],
            },
        },
        "Targeting_Efficiency": {
            TOO_LOW: {
                "recommendation": "Increase ligand (folate/antibody) conjugation density during post-synthesis functionalization.",
                "mechanism": "Higher receptor-binding ligand density improves selective cellular uptake.",
                "adjusts": [("targeting_ligand", "increase")],
            },
        },
        "Zeta_Potential_mV": {
            TOO_HIGH: {  # too positive, CTAB residue
                "recommendation": "Perform a CTAB-to-PEG-thiol ligand exchange.",
                "mechanism": "Replacing the cationic CTAB bilayer with a neutral/anionic "
                             "PEG-thiol layer removes the excess positive surface charge.",
                "adjusts": [("CTAB", "decrease"), ("PEG", "add")],
            },
        },
        "Drug_Loading_Efficiency": {
            TOO_LOW: {
                "recommendation": "Replace residual CTAB with a mixed PEG-thiol / drug-carrier monolayer via ligand exchange.",
                "mechanism": "The dense CTAB bilayer blocks drug access; exchanging it frees "
                             "surface sites for conjugation.",
                "adjusts": [("CTAB", "decrease"), ("ligand", "add")],
            },
        },
        "Cytotoxicity": {
            TOO_HIGH: {
                "recommendation": "Perform a CTAB-to-PEG-thiol ligand exchange and add a wash step.",
                "mechanism": "Free CTAB is strongly cytotoxic; removing it lowers off-target toxicity.",
                "adjusts": [("CTAB", "decrease"), ("wash", "add")],
            },
        },
        "PDI": {
            TOO_HIGH: {
                "recommendation": "Use a more monodisperse seed batch and add seeds rapidly to a well-mixed growth solution.",
                "mechanism": "Uniform seeds and fast, uniform initiation reduce variation during overgrowth.",
                "adjusts": [("seed", "standardize")],
            },
        },
    },

    BRUST: {
        "Particle_Size_nm": {
            TOO_HIGH: {
                "recommendation": "Increase the thiol-to-gold ratio.",
                "mechanism": "Higher capping-ligand availability terminates growth earlier "
                             "(Brust's original finding), giving sub-5 nm cores.",
                "adjusts": [("thiol", "increase")],
            },
            TOO_LOW: {
                "recommendation": "Decrease the thiol-to-gold ratio, or slow the reduction (less NaBH4).",
                "mechanism": "Less capping and slower reduction allow more growth per nucleus.",
                "adjusts": [("thiol", "decrease"), ("NaBH4", "decrease")],
            },
        },
        "Particle_Width_nm": {
            TOO_HIGH: {
                "recommendation": "Increase the thiol-to-gold ratio.",
                "mechanism": "Brust particles are small and near-spherical, so width tracks "
                             "diameter; more thiol caps growth earlier.",
                "adjusts": [("thiol", "increase")],
            },
        },
        "Drug_Loading_Efficiency": {
            TOO_LOW: {
                "recommendation": "Use a longer-chain or mixed-monolayer thiol ligand shell.",
                "mechanism": "Increases the available functional surface density for drug conjugation.",
                "adjusts": [("thiol", "modify"), ("ligand", "add")],
            },
        },
        "Cytotoxicity": {
            TOO_HIGH: {
                "recommendation": "Reduce excess free thiol and add a purification/wash step.",
                "mechanism": "Unbound thiol byproducts are a known cytotoxicity contributor, "
                             "separate from the AuNP core itself.",
                "adjusts": [("thiol", "decrease"), ("wash", "add")],
            },
        },
        "Targeting_Efficiency": {
            TOO_LOW: {
                "recommendation": "Introduce targeting ligands into the thiol monolayer via place-exchange.",
                "mechanism": "Thiol place-exchange lets receptor-binding ligands be added at controlled density.",
                "adjusts": [("targeting_ligand", "add")],
            },
        },
        "PDI": {
            TOO_HIGH: {
                "recommendation": "Cool the reaction and add NaBH4 rapidly to a vigorously stirred biphasic mixture.",
                "mechanism": "Fast, uniform reduction at low temperature narrows the size distribution.",
                "adjusts": [("temperature", "decrease"), ("NaBH4", "increase")],
            },
        },
        "Zeta_Potential_mV": {
            TOO_HIGH: {
                "recommendation": "Use a thiol with an anionic terminal group (e.g. a carboxylate-terminated thiol).",
                "mechanism": "Anionic end-groups impart a more negative surface charge.",
                "adjusts": [("thiol", "modify")],
            },
        },
    },

    GREEN: {
        "Particle_Size_nm": {
            TOO_HIGH: {
                "recommendation": "Increase plant-extract (bioreductant) concentration or raise the reaction temperature.",
                "mechanism": "More bioreductant accelerates reduction kinetics, favouring more "
                             "nuclei and therefore smaller particles.",
                "adjusts": [("extract", "increase"), ("temperature", "increase")],
            },
            TOO_LOW: {
                "recommendation": "Decrease extract concentration or lower the reaction temperature.",
                "mechanism": "Slower reduction produces fewer nuclei and larger particles.",
                "adjusts": [("extract", "decrease"), ("temperature", "decrease")],
            },
        },
        "Particle_Width_nm": {
            TOO_HIGH: {
                "recommendation": "Increase plant-extract concentration or raise the reaction temperature.",
                "mechanism": "Faster reduction gives more, smaller, near-isotropic particles.",
                "adjusts": [("extract", "increase"), ("temperature", "increase")],
            },
        },
        "PDI": {
            TOO_HIGH: {
                "recommendation": "Standardize / filter the extract and increase the extract-to-precursor ratio.",
                "mechanism": "Reduces the compositional variability inherent to natural extracts, "
                             "improving batch-to-batch uniformity.",
                "adjusts": [("extract", "standardize"), ("extract", "increase")],
            },
        },
        "Drug_Loading_Efficiency": {
            TOO_LOW: {
                "recommendation": "Add a post-synthesis ligand-exchange step.",
                "mechanism": "The native protein/polyphenol capping layer from the extract can "
                             "occlude drug-binding sites; exchanging it frees surface area.",
                "adjusts": [("ligand", "add")],
            },
        },
        "Targeting_Efficiency": {
            TOO_LOW: {
                "recommendation": "Conjugate targeting ligands after partly displacing the native biomolecular corona.",
                "mechanism": "The extract-derived corona must be partly removed to present "
                             "receptor-binding ligands.",
                "adjusts": [("targeting_ligand", "add")],
            },
        },
        "Zeta_Potential_mV": {
            TOO_HIGH: {
                "recommendation": "Raise the pH or increase polyphenol-rich extract content.",
                "mechanism": "Deprotonated polyphenol/carboxyl groups increase the negative surface charge.",
                "adjusts": [("pH", "increase"), ("extract", "increase")],
            },
        },
        "Cytotoxicity": {
            TOO_HIGH: {
                "recommendation": "Purify to remove residual extract and unreduced precursor.",
                "mechanism": "Residual reactive extract components and free gold ions contribute "
                             "off-target toxicity.",
                "adjusts": [("wash", "add")],
            },
        },
    },
}


# ===========================================================================
# 2. METHOD INFERENCE  --  reducer/stabilizer/morphology -> method
# ===========================================================================
# The dataset has no explicit "method" column, but the reducer is a clean proxy for the
# canonical synthesis route. Stabilizer and morphology act as fall-backs / tie-breakers.

_REDUCER_TO_METHOD = {
    "citrate": TURKEVICH,
    "ascorbic_acid": SEED_MEDIATED,
    "ascorbic acid": SEED_MEDIATED,
    "nabh4": BRUST,
    "ethylene_glycol": GREEN,
    "ethylene glycol": GREEN,
}

_STABILIZER_TO_METHOD = {
    "citrate": TURKEVICH,
    "ctab": SEED_MEDIATED,
    "toab": BRUST,
    "pvp": GREEN,
}


def _norm(value: Any) -> str:
    """Lower-case, strip, collapse spaces - tolerant of dataset label variants."""
    return str(value).strip().lower() if value is not None else ""


def infer_synthesis_method(user_input: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Infer the synthesis method from the parameters the user selected.

    Args:
        user_input: mapping of feature name -> value (e.g. the Streamlit ``user_input`` dict
                    or a ``pd.Series``). Looks for ``Reducer``, then ``Stabilizer``.

    Returns:
        (method, basis) where ``method`` is one of the canonical names or ``None`` if it
        cannot be determined, and ``basis`` is a short human-readable explanation.
    """
    # Accept dicts and pandas Series alike.
    keys = list(user_input.keys()) if hasattr(user_input, "keys") else list(getattr(user_input, "index", []))
    get = user_input.get if hasattr(user_input, "get") else lambda k, d=None: user_input[k] if k in user_input else d

    # 1. Plain categorical column (Streamlit UI passes e.g. Reducer="citrate").
    reducer = _norm(get("Reducer"))
    if reducer in _REDUCER_TO_METHOD:
        return _REDUCER_TO_METHOD[reducer], f"inferred from {reducer.replace('_', ' ')} reduction"

    stabilizer = _norm(get("Stabilizer"))
    if stabilizer in _STABILIZER_TO_METHOD:
        return _STABILIZER_TO_METHOD[stabilizer], f"inferred from {stabilizer.upper()} stabilizer"

    # 2. One-hot encoded dataset (transformed CSV: e.g. Reducer_citrate=1).
    onehot = _method_from_onehot(keys, get, "Reducer_", _REDUCER_TO_METHOD, "reduction")
    if onehot[0]:
        return onehot
    onehot = _method_from_onehot(keys, get, "Stabilizer_", _STABILIZER_TO_METHOD, "stabilizer")
    if onehot[0]:
        return onehot

    return None, "could not be determined from the selected reducer/stabilizer"


def _method_from_onehot(keys, get, prefix, table, noun) -> Tuple[Optional[str], str]:
    """Find an active one-hot column (prefix_<value> == 1) and map its value to a method."""
    for key in keys:
        if not str(key).startswith(prefix):
            continue
        try:
            active = float(get(key)) >= 0.5
        except (TypeError, ValueError):
            active = bool(get(key))
        if not active:
            continue
        value = _norm(str(key)[len(prefix):])
        if value in table:
            return table[value], f"inferred from {value.replace('_', ' ')} {noun}"
    return None, ""


# ===========================================================================
# 3. DEVIATION SCORE (Delta_k, Eq. 16)
# ===========================================================================

def compute_deviations(
    predictions: Dict[str, float],
    criteria: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compute the Delta_k deviation for every predicted property against its ideal range.

    A property in range contributes nothing. Out of range, it records the direction and the
    raw gap (in the property's own units) plus a range-normalized magnitude used to rank
    deviations of different properties on a comparable scale.

    Returns:
        List of dicts sorted by descending normalized magnitude, each with keys:
        property, value, low, high, direction, gap, normalized, unit.
    """
    deviations: List[Dict[str, Any]] = []
    for prop, spec in criteria.items():
        if prop not in predictions:
            continue
        low, high = spec["ideal_range"]
        value = float(predictions[prop])
        width = float(high - low) if high != low else 1.0

        direction = None
        gap = 0.0
        if value < low:
            direction, gap = TOO_LOW, low - value
        elif value > high:
            direction, gap = TOO_HIGH, value - high
        else:
            continue  # in range

        canon = canonical_property(prop)
        deviations.append({
            "property": prop,           # original label, used for display
            "lookup_key": canon,        # canonical key, used for rule lookup
            "value": value,
            "low": low,
            "high": high,
            "direction": direction,
            "gap": gap,
            "normalized": gap / width,  # Delta_k magnitude, comparable across properties
            "unit": PROPERTY_UNITS.get(canon, ""),
        })

    deviations.sort(key=lambda d: d["normalized"], reverse=True)
    return deviations


# ===========================================================================
# 4. LOOKUP + REPORT ASSEMBLY
# ===========================================================================

def lookup_recommendation(method: str, prop: str, direction: str) -> Optional[Dict[str, Any]]:
    """Deterministic (method, property, direction) -> recommendation dict, or None."""
    return SYNTHESIS_METHOD_RULES.get(method, {}).get(prop, {}).get(direction)


def _prop_label(prop: str) -> str:
    return (prop.replace("_nm", "").replace("_mV", "").replace("_%", "")
            .replace("_", " ").strip().title())


def _format_gap(dev: Dict[str, Any]) -> str:
    unit = f" {dev['unit']}" if dev["unit"] else ""
    where = "above" if dev["direction"] == TOO_HIGH else "below"
    return (f"predicted {dev['value']:.2f}{unit}, "
            f"{dev['gap']:.2f}{unit} {where} the ideal range "
            f"({dev['low']} to {dev['high']}{unit})")


def _synergy_note(applied: List[Dict[str, Any]]) -> Optional[str]:
    """
    Detect whether the applied recommendations reinforce or conflict.

    Two recommendations reinforce if they adjust the same reagent in the same direction, and
    conflict if they adjust it in opposite directions. This is the one piece of light
    "reasoning" the design keeps - computed deterministically from set logic, no model call.
    """
    # reagent -> set of directions requested across the applied recommendations
    reagent_dirs: Dict[str, set] = {}
    for item in applied:
        for reagent, direction in item["rule"].get("adjusts", []):
            reagent_dirs.setdefault(reagent, set()).add(direction)

    reinforce: List[str] = []
    conflict: List[str] = []
    for reagent, dirs in reagent_dirs.items():
        label = REAGENT_LABELS.get(reagent, reagent.replace("_", " "))
        opposing = ({"increase", "decrease"} <= dirs)
        if opposing:
            conflict.append(label)
        elif len(dirs) == 1:
            # Only meaningful as a synergy if >1 recommendation touches this reagent.
            count = sum(1 for it in applied
                        for r, _ in it["rule"].get("adjusts", []) if r == reagent)
            if count >= 2:
                gerund = _DIRECTION_GERUND.get(next(iter(dirs)), next(iter(dirs)))
                reinforce.append(f"{gerund} {label}")

    parts: List[str] = []
    if reinforce:
        joined = "; ".join(sorted(set(reinforce)))
        parts.append(f"Several fixes point the same way ({joined}), so a single change "
                     f"addresses more than one deviation.")
    if conflict:
        joined = "; ".join(sorted(set(conflict)))
        parts.append(f"Caution: recommendations place opposing demands on {joined} - these "
                     f"trade off and must be balanced experimentally.")
    return " ".join(parts) if parts else None


def _reconcile_banner(success_prediction: Optional[bool], has_deviations: bool) -> Optional[str]:
    """
    Reconcile the binary success classifier with the range-based deviation check so the
    report never contradicts the model. The two heads can disagree because the classifier
    captures parameter interactions that per-property ranges do not.
    """
    if success_prediction is None:
        return None
    if success_prediction:  # model predicts SUCCESS
        if has_deviations:
            return ("The model predicts a **successful** synthesis; the items below are "
                    "refinements that move borderline properties further into the ideal window.")
        return "The model predicts a **successful** synthesis and every property is in range."
    # model predicts UNSUCCESSFUL
    if has_deviations:
        return ("**Note —** the model predicts this synthesis is **unsuccessful**, consistent "
                "with the out-of-range properties addressed below.")
    return ("**Note —** the model predicts this synthesis is **unsuccessful even though every "
            "property is within range**. In-range values are necessary but not sufficient: the "
            "success classifier weighs parameter interactions the per-property ranges don't "
            "capture. There is no single property to target — revisit the synthesis route and "
            "conditions holistically rather than tuning one output.")


def build_recommendation_report(
    user_input: Dict[str, Any],
    predictions: Dict[str, float],
    criteria: Dict[str, Dict[str, Any]],
    use_case: str = "",
    max_items: int = 3,
    success_prediction: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Assemble the full deterministic recommendation report.

    Args:
        success_prediction: optional binary verdict from the model's classification head.
            When provided, the report reconciles it with the range check (see
            ``_reconcile_banner``) so the recommendations never silently contradict the model.

    Returns a dict with:
        method      : inferred method (or None)
        method_basis: why that method was inferred
        deviations  : ranked deviation list (from compute_deviations)
        applied     : the recommendations actually shown (top ``max_items`` with a rule)
        unmatched   : deviations that had no pre-validated rule
        markdown    : a ready-to-render Markdown report
    """
    method, basis = infer_synthesis_method(user_input)
    deviations = compute_deviations(predictions, criteria)

    lines: List[str] = []
    case_suffix = f" for **{use_case}**" if use_case else ""

    if method:
        lines.append(f"**Detected synthesis method:** {method}  _({basis})_")
    else:
        lines.append(f"**Synthesis method:** _{basis}; showing range check only._")
    lines.append("")

    banner = _reconcile_banner(success_prediction, has_deviations=bool(deviations))

    if not deviations:
        if banner:
            lines.append(banner)
        else:
            lines.append(f"All predicted properties fall within the ideal range{case_suffix}. "
                         f"No synthesis adjustments are required.")
        return {
            "method": method, "method_basis": basis, "use_case": use_case,
            "deviations": [], "applied": [], "unmatched": [],
            "success_prediction": success_prediction,
            "markdown": "\n".join(lines).strip(),
        }

    if banner:
        lines.append(banner)
        lines.append("")
    lines.append(f"**Pre-validated optimization recommendations**{case_suffix} "
                 f"(ranked by deviation severity, Δₖ):")
    lines.append("")

    applied: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []

    for dev in deviations[:max_items]:
        rule = lookup_recommendation(method, dev["lookup_key"], dev["direction"]) if method else None
        label = _prop_label(dev["property"])
        if rule:
            applied.append({"dev": dev, "rule": rule})
            idx = len(applied)
            lines.append(f"{idx}. **{label}** — {_format_gap(dev)}.")
            lines.append(f"   - **Action:** {rule['recommendation']}")
            lines.append(f"   - **Why:** {rule['mechanism']}")
        else:
            unmatched.append(dev)
            human_dir = "too high" if dev["direction"] == TOO_HIGH else "too low"
            method_name = method or "unknown method"
            lines.append(f"- **{label}** — {_format_gap(dev)}. "
                         f"_No pre-validated rule for ({method_name}, {label}, {human_dir}); "
                         f"flagged for chemist review._")
        lines.append("")

    note = _synergy_note(applied)
    if note:
        lines.append(f"> **Synergy:** {note}")
        lines.append("")

    # Mention any further out-of-range properties beyond max_items, without recommendations.
    extra = deviations[max_items:]
    if extra:
        names = ", ".join(_prop_label(d["property"]) for d in extra)
        lines.append(f"_Also out of range (lower severity): {names}._")

    return {
        "method": method,
        "method_basis": basis,
        "use_case": use_case,
        "deviations": deviations,
        "applied": applied,
        "unmatched": unmatched,
        "success_prediction": success_prediction,
        "markdown": "\n".join(lines).strip(),
    }


# ===========================================================================
# 5. DETAILED EXPLANATION LAYER  (optional, grounded in the rule table)
# ===========================================================================

def _explanation_prompt(report: Dict[str, Any]) -> str:
    """Build a grounded prompt that asks for a proper teaching-quality explanation of the
    pre-approved recommendations. The actionable chemistry is fixed by the rule table; the
    explanation layer may only deepen the reasoning around it, never replace it."""
    method = report.get("method") or "the inferred synthesis route"
    use_case = report.get("use_case") or "the selected application"
    succ = report.get("success_prediction")
    verdict = ("the model predicts this synthesis will be SUCCESSFUL" if succ
               else "the model predicts this synthesis will be UNSUCCESSFUL" if succ is False
               else "no overall success verdict is available")

    # Authoritative findings, passed as structured facts the explanation must respect.
    facts: List[str] = []
    for item in report.get("applied", []):
        dev, rule = item["dev"], item["rule"]
        unit = f" {dev['unit']}" if dev["unit"] else ""
        where = "above" if dev["direction"] == TOO_HIGH else "below"
        facts.append(
            f"- Property '{_prop_label(dev['property'])}': predicted {dev['value']:.2f}{unit}, "
            f"{dev['gap']:.2f}{unit} {where} the ideal range "
            f"({dev['low']} to {dev['high']}{unit}).\n"
            f"    APPROVED ACTION: {rule['recommendation']}\n"
            f"    APPROVED MECHANISM (authoritative): {rule['mechanism']}"
        )
    for dev in report.get("unmatched", []):
        facts.append(f"- Property '{_prop_label(dev['property'])}' is out of range but has no "
                     f"pre-approved rule for this method; note it needs manual review, do not invent a fix.")
    facts_block = "\n".join(facts) if facts else "- All predicted properties are within the ideal range."

    return (
        "You are an expert in gold-nanoparticle (AuNP) synthesis writing a clear, instructive "
        "explanation for a researcher who will run the experiment. Explain the findings below "
        "properly — do not merely restate them.\n\n"
        f"CONTEXT\n- Synthesis route: {method}\n- Target application: {use_case}\n- {verdict}.\n\n"
        f"FINDINGS AND APPROVED ACTIONS (authoritative — these are fixed):\n{facts_block}\n\n"
        "WRITE the explanation with these sections:\n"
        "1. **What these results mean** — 2-3 sentences interpreting the overall picture for the "
        "target application, referring to the success verdict.\n"
        "2. **Each adjustment, explained** — for every approved action, a short paragraph that "
        "(a) deepens the approved mechanism using established AuNP synthesis chemistry, and "
        "(b) gives concrete, practical lab guidance on how to make the change and what to watch for.\n"
        "3. **Sequencing** — if the actions reinforce or conflict, explain a sensible order to try them.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY the approved actions above. Do NOT invent new actions, reagents, numbers, or properties.\n"
        "- Each 'APPROVED MECHANISM' is authoritative; you may expand on it with standard textbook "
        "chemistry but must never contradict it or the approved action.\n"
        "- Add genuine explanatory and practical value; never just repeat the bullets verbatim.\n"
        "- Be precise and educational; avoid filler and marketing language."
    )


def generate_explanation(report: Dict[str, Any], text_generator) -> str:
    """
    Produce an in-depth, grounded explanation of a recommendation report.

    The actionable chemistry (what to change) is fixed by the deterministic rule table; this
    layer only elaborates the *reasoning* and adds practical lab guidance, under strict
    grounding constraints (see ``_explanation_prompt``). The structured rule output remains
    the system of record and is always shown alongside this prose.

    Args:
        report:          the dict returned by ``build_recommendation_report``.
        text_generator:  a callable ``str -> str`` that turns a prompt into text.
    """
    return text_generator(_explanation_prompt(report))


# ===========================================================================
# Self-test  --  run `python recommender.py` to verify logic without Streamlit.
# ===========================================================================
if __name__ == "__main__":
    demo_criteria = {
        "Particle_Size_nm": {"ideal_range": (10, 150), "description": ""},
        "Particle_Width_nm": {"ideal_range": (5, 40), "description": ""},
        "Zeta_Potential_mV": {"ideal_range": (-30, -5), "description": ""},
        "Drug_Loading_Efficiency": {"ideal_range": (30, 60), "description": ""},
        "Targeting_Efficiency": {"ideal_range": (40, 60), "description": ""},
        "Cytotoxicity": {"ideal_range": (10, 30), "description": ""},
    }

    print("=" * 70)
    print("CASE A: Turkevich - size too large AND zeta not negative enough")
    print("        (both fixes call for more citrate -> synergy note should fire)")
    print("=" * 70)
    user_a = {"Reducer": "citrate", "Stabilizer": "citrate", "Morphology": "nanosphere"}
    pred_a = {
        "Particle_Size_nm": 185.0,   # too high  -> increase citrate
        "Particle_Width_nm": 22.0,   # in range
        "Zeta_Potential_mV": -2.0,   # too high (not negative enough) -> increase citrate
        "Drug_Loading_Efficiency": 50.0,  # in range
        "Targeting_Efficiency": 50.0,     # in range
        "Cytotoxicity": 20.0,             # in range
    }
    print(build_recommendation_report(user_a, pred_a, demo_criteria, "Cancer Treatment (General)")["markdown"])

    print("\n" + "=" * 70)
    print("CASE B: Seed-Mediated, everything in range")
    print("=" * 70)
    user_b = {"Reducer": "ascorbic_acid", "Stabilizer": "CTAB", "Morphology": "nanorod"}
    pred_b = {k: sum(v["ideal_range"]) / 2 for k, v in demo_criteria.items()}
    print(build_recommendation_report(user_b, pred_b, demo_criteria)["markdown"])

    print("\n" + "=" * 70)
    print("CASE C: Unknown method (graceful) + an unmatched property")
    print("=" * 70)
    user_c = {"Reducer": "mystery_reductant"}
    pred_c = {"Particle_Size_nm": 200.0, "Cytotoxicity": 5.0}
    print(build_recommendation_report(user_c, pred_c, demo_criteria)["markdown"])

    print("\n" + "=" * 70)
    print("CASE D: RECONCILIATION - all properties in range, but model predicts UNSUCCESSFUL")
    print("        (the screenshot scenario: must NOT claim 'no adjustments required')")
    print("=" * 70)
    user_d = {"Reducer": "citrate", "Stabilizer": "citrate"}
    pred_d = {k: sum(v["ideal_range"]) / 2 for k, v in demo_criteria.items()}  # all mid-range
    print(build_recommendation_report(user_d, pred_d, demo_criteria,
                                      "Cancer Treatment (General)",
                                      success_prediction=False)["markdown"])

    print("\n" + "=" * 70)
    print("CASE E: RECONCILIATION - deviations present AND model predicts UNSUCCESSFUL")
    print("=" * 70)
    print(build_recommendation_report(user_a, pred_a, demo_criteria,
                                      "Cancer Treatment (General)",
                                      success_prediction=False)["markdown"])
