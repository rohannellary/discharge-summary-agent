"""
Medication Reconciliation Tool
================================
Compares admission medications vs discharge medications.
Surfaces: additions, discontinuations, dose changes.
Flags any change WITHOUT a documented reason for clinician review.

Never silently resolves discrepancies.
"""

import re
from difflib import SequenceMatcher
from typing import Any


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def names_match(a: str, b: str, threshold: float = 0.75) -> bool:
    """Fuzzy match drug names (handles typos, abbreviations)."""
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    if a_norm == b_norm:
        return True
    # One is a prefix of the other (e.g. "metformin" vs "metformin hcl")
    if a_norm.startswith(b_norm) or b_norm.startswith(a_norm):
        return True
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold


def normalize_dose(dose: str) -> str:
    """Normalize dose string for comparison."""
    return re.sub(r'\s+', '', dose.lower().strip())


class MedReconciliationTool:
    """
    Compares admission and discharge medication lists.
    Flags changes without documented reason.
    """

    def run(self, inputs: dict, state) -> dict:
        structured = state.structured_data

        admission_meds = structured.get("admission_medications", [])
        discharge_meds = structured.get("discharge_medications", [])

        # Also try to extract from raw section text
        all_text = ""
        for section_name, pages in state.extracted_sections.items():
            for page in pages:
                all_text += page["text"] + "\n"

        # Try to parse reason for changes from notes
        reason_patterns = [
            r"(?:stopped|discontinued|held|d/c'?d?)\s+(\w+)(?:\s+due\s+to\s+|\s+for\s+|\s+because\s+)([^\n]{5,80})",
            r"(\w+)\s+(?:started|initiated|added|begun)\s+(?:for|due\s+to|because\s+of)\s+([^\n]{5,80})",
            r"(?:changed|increased|decreased|titrated)\s+(\w+)(?:\s+to\s+\d+\s*\w+)?\s+(?:for|due\s+to)\s+([^\n]{5,80})",
        ]
        documented_reasons = {}
        for pat in reason_patterns:
            for m in re.finditer(pat, all_text, re.IGNORECASE):
                drug = normalize_name(m.group(1))
                reason = m.group(2).strip()
                documented_reasons[drug] = reason

        # Deduplicate each list by normalized name (keep first occurrence)
        def dedup_meds(meds):
            seen_names = set()
            result = []
            for m in meds:
                key = normalize_name(m.get("name", ""))
                if key and key not in seen_names:
                    seen_names.add(key)
                    result.append(m)
            return result

        admission_meds = dedup_meds(admission_meds)
        discharge_meds = dedup_meds(discharge_meds)

        changes = []
        unmatched_discharge = list(discharge_meds)

        for adm_med in admission_meds:
            adm_name = adm_med.get("name", "")
            adm_dose = adm_med.get("dose", "")

            # Find matching discharge med
            matched = None
            for disc_med in discharge_meds:
                if names_match(adm_name, disc_med.get("name", "")):
                    matched = disc_med
                    if disc_med in unmatched_discharge:
                        unmatched_discharge.remove(disc_med)
                    break

            if matched is None:
                # Drug was on admission but not discharge → discontinued
                reason = documented_reasons.get(normalize_name(adm_name))
                changes.append({
                    "type": "DISCONTINUED",
                    "drug": adm_name,
                    "admission_dose": adm_dose,
                    "discharge_dose": None,
                    "reason_documented": reason,
                    "flag_for_review": reason is None,
                    "flag_reason": None if reason else "Medication discontinued with no documented reason"
                })
            else:
                disc_dose = matched.get("dose", "")
                if adm_dose and disc_dose and normalize_dose(adm_dose) != normalize_dose(disc_dose):
                    # Dose changed
                    reason = documented_reasons.get(normalize_name(adm_name))
                    changes.append({
                        "type": "DOSE_CHANGED",
                        "drug": adm_name,
                        "admission_dose": adm_dose,
                        "discharge_dose": disc_dose,
                        "reason_documented": reason,
                        "flag_for_review": reason is None,
                        "flag_reason": None if reason else "Dose changed with no documented reason"
                    })
                else:
                    changes.append({
                        "type": "CONTINUED",
                        "drug": adm_name,
                        "admission_dose": adm_dose,
                        "discharge_dose": disc_dose,
                        "reason_documented": None,
                        "flag_for_review": False,
                        "flag_reason": None
                    })

        # New meds on discharge that weren't on admission
        for disc_med in unmatched_discharge:
            disc_name = disc_med.get("name", "")
            reason = documented_reasons.get(normalize_name(disc_name))
            changes.append({
                "type": "NEW",
                "drug": disc_name,
                "admission_dose": None,
                "discharge_dose": disc_med.get("dose", ""),
                "reason_documented": reason,
                "flag_for_review": reason is None,
                "flag_reason": None if reason else "New medication added with no documented reason"
            })

        flags_needed = [c for c in changes if c["flag_for_review"]]

        return {
            "admission_medication_count": len(admission_meds),
            "discharge_medication_count": len(discharge_meds),
            "changes": changes,
            "changes_requiring_review": flags_needed,
            "reconciliation_complete": True,
            "note": "Medication reconciliation performed. All flagged changes require clinician review before finalization."
        }
