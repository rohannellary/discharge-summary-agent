"""
Drug Interaction Tool
======================
Mocked drug-interaction lookup — in production this would call an external API
(e.g. DrugBank, OpenFDA, Lexicomp).

Simulates realistic interactions for common drug pairs.
If an interaction is found it is ALWAYS surfaced and escalated — never buried.
"""

import re
from typing import Any

# ── Mock interaction database ──────────────────────────────────────────────────
# Format: frozenset({drug_a, drug_b}) → interaction_info
INTERACTION_DB = {
    frozenset({"warfarin", "aspirin"}): {
        "severity": "MAJOR",
        "description": "Concurrent use increases bleeding risk significantly. Monitor INR closely.",
        "clinical_significance": "HIGH",
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "MAJOR",
        "description": "NSAIDs inhibit platelet function and may displace warfarin from protein binding, increasing hemorrhage risk.",
        "clinical_significance": "HIGH",
    },
    frozenset({"metformin", "contrast"}): {
        "severity": "MODERATE",
        "description": "Iodinated contrast may cause acute kidney injury, increasing risk of metformin-associated lactic acidosis.",
        "clinical_significance": "MODERATE",
    },
    frozenset({"lisinopril", "potassium"}): {
        "severity": "MODERATE",
        "description": "ACE inhibitors reduce potassium excretion; concurrent potassium supplementation may cause hyperkalemia.",
        "clinical_significance": "MODERATE",
    },
    frozenset({"lisinopril", "spironolactone"}): {
        "severity": "MODERATE",
        "description": "Combined use of ACE inhibitor and potassium-sparing diuretic increases hyperkalemia risk.",
        "clinical_significance": "MODERATE",
    },
    frozenset({"ciprofloxacin", "warfarin"}): {
        "severity": "MAJOR",
        "description": "Fluoroquinolones inhibit CYP1A2, increasing warfarin levels and bleeding risk.",
        "clinical_significance": "HIGH",
    },
    frozenset({"metoprolol", "verapamil"}): {
        "severity": "MAJOR",
        "description": "Additive negative chronotropic and inotropic effects — risk of complete heart block.",
        "clinical_significance": "HIGH",
    },
    frozenset({"ssri", "tramadol"}): {
        "severity": "MAJOR",
        "description": "Risk of serotonin syndrome. Both increase serotonergic activity.",
        "clinical_significance": "HIGH",
    },
    frozenset({"sertraline", "tramadol"}): {
        "severity": "MAJOR",
        "description": "Risk of serotonin syndrome. Both increase serotonergic activity.",
        "clinical_significance": "HIGH",
    },
    frozenset({"fluoxetine", "tramadol"}): {
        "severity": "MAJOR",
        "description": "Risk of serotonin syndrome. Fluoxetine also inhibits CYP2D6 reducing tramadol conversion.",
        "clinical_significance": "HIGH",
    },
    frozenset({"amiodarone", "warfarin"}): {
        "severity": "MAJOR",
        "description": "Amiodarone inhibits CYP2C9 and CYP3A4, substantially increasing warfarin levels.",
        "clinical_significance": "HIGH",
    },
    frozenset({"digoxin", "amiodarone"}): {
        "severity": "MAJOR",
        "description": "Amiodarone increases digoxin levels by ~70%. Risk of digoxin toxicity.",
        "clinical_significance": "HIGH",
    },
    frozenset({"lithium", "ibuprofen"}): {
        "severity": "MAJOR",
        "description": "NSAIDs reduce renal lithium clearance, causing toxicity.",
        "clinical_significance": "HIGH",
    },
    frozenset({"insulin", "metformin"}): {
        "severity": "MINOR",
        "description": "Additive hypoglycemic effect. Monitor blood glucose.",
        "clinical_significance": "LOW",
    },
    frozenset({"furosemide", "gentamicin"}): {
        "severity": "MAJOR",
        "description": "Additive ototoxicity and nephrotoxicity risk.",
        "clinical_significance": "HIGH",
    },
    frozenset({"simvastatin", "amiodarone"}): {
        "severity": "MAJOR",
        "description": "Amiodarone inhibits CYP3A4 — simvastatin dose should not exceed 20mg to avoid myopathy/rhabdomyolysis.",
        "clinical_significance": "HIGH",
    },
    frozenset({"clopidogrel", "omeprazole"}): {
        "severity": "MODERATE",
        "description": "Omeprazole inhibits CYP2C19 reducing clopidogrel activation. Consider alternative PPI.",
        "clinical_significance": "MODERATE",
    },
    frozenset({"methotrexate", "nsaids"}): {
        "severity": "MAJOR",
        "description": "NSAIDs reduce methotrexate renal clearance, causing toxicity.",
        "clinical_significance": "HIGH",
    },
}

# Drug name aliases / generic→brand normalisation
ALIASES = {
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "aspirin": "aspirin",
    "coumadin": "warfarin",
    "glucophage": "metformin",
    "zocor": "simvastatin",
    "lipitor": "atorvastatin",
    "plavix": "clopidogrel",
    "lasix": "furosemide",
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "cordarone": "amiodarone",
    "lanoxin": "digoxin",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "aldactone": "spironolactone",
    "prilosec": "omeprazole",
    "nexium": "esomeprazole",
}

SSRI_NAMES = {"sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
              "fluvoxamine", "venlafaxine", "duloxetine"}
NSAID_NAMES = {"ibuprofen", "naproxen", "diclofenac", "ketorolac", "indomethacin",
               "celecoxib", "meloxicam"}


def normalize_drug(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r'\s+', ' ', n)
    return ALIASES.get(n, n)


def expand_class(name: str) -> list[str]:
    """Expand drug class names to canonical names for lookup."""
    n = name.lower()
    expanded = [n]
    if n in SSRI_NAMES:
        expanded.append("ssri")
    if n in NSAID_NAMES:
        expanded.append("nsaids")
    return expanded


class DrugInteractionTool:
    """
    Checks discharge medication list for significant interactions.
    Any finding is surfaced — never buried.
    """

    def run(self, inputs: dict, state) -> dict:
        structured = state.structured_data
        discharge_meds = structured.get("discharge_medications", [])

        if not discharge_meds:
            return {
                "interactions": [],
                "checked": False,
                "note": "No discharge medications available to check."
            }

        drug_names = [normalize_drug(m.get("name", "")) for m in discharge_meds if m.get("name")]
        drug_names = [d for d in drug_names if len(d) >= 3]

        interactions_found = []
        checked_pairs = set()

        for i, drug_a in enumerate(drug_names):
            for drug_b in drug_names[i + 1:]:
                pair = frozenset({drug_a, drug_b})
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Expand to class names
                a_variants = expand_class(drug_a)
                b_variants = expand_class(drug_b)

                for av in a_variants:
                    for bv in b_variants:
                        lookup_pair = frozenset({av, bv})
                        if lookup_pair in INTERACTION_DB:
                            info = INTERACTION_DB[lookup_pair]
                            interactions_found.append({
                                "drug_a": drug_a,
                                "drug_b": drug_b,
                                "severity": info["severity"],
                                "description": info["description"],
                                "clinical_significance": info["clinical_significance"],
                                "requires_clinician_review": info["severity"] in ("MAJOR", "MODERATE"),
                                "flag_message": f"{info['severity']} interaction between {drug_a} and {drug_b}: {info['description']}"
                            })

        # Deduplicate
        seen = set()
        unique = []
        for ix in interactions_found:
            key = (ix["drug_a"], ix["drug_b"])
            if key not in seen:
                seen.add(key)
                unique.append(ix)

        major = [i for i in unique if i["severity"] == "MAJOR"]
        moderate = [i for i in unique if i["severity"] == "MODERATE"]
        minor = [i for i in unique if i["severity"] == "MINOR"]

        return {
            "interactions": unique,
            "major_count": len(major),
            "moderate_count": len(moderate),
            "minor_count": len(minor),
            "checked": True,
            "drugs_checked": drug_names,
            "note": f"Checked {len(checked_pairs)} drug pairs. Found {len(unique)} interactions ({len(major)} major, {len(moderate)} moderate)."
        }
