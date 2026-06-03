"""
Conflict Detector Tool
=======================
Finds contradictions between notes — different diagnoses, different medication
doses, different dates — and flags them for clinician review.

Never resolves conflicts automatically. Always surfaces them.
"""

import re
from difflib import SequenceMatcher
from typing import Any


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class ConflictDetectorTool:
    """
    Scans extracted data for contradictions between documents.
    Returns list of conflicts, each with source attribution.
    """

    def run(self, inputs: dict, state) -> dict:
        conflicts = []
        structured = state.structured_data
        sections = state.extracted_sections

        # ── 1. Diagnosis conflicts ────────────────────────────────────────────
        diagnosis_mentions = []
        for section_name, pages in sections.items():
            for page in pages:
                text = page["text"]
                source = f"{page['source']} p{page['page']}"

                # Find principal diagnosis mentions
                for pat in [
                    r"(?:principal|primary|discharge|final|admitting)\s*diagnosis[:\s]+([^\n]{5,100})",
                ]:
                    for m in re.finditer(pat, text, re.IGNORECASE):
                        dx = m.group(1).strip()
                        if len(dx) > 4:
                            diagnosis_mentions.append({"source": source, "diagnosis": dx})

        # Compare pairs for conflicts
        # Only flag genuine cross-document conflicts; cap at 5 to avoid noise
        seen_conflict_pairs = set()
        for i in range(len(diagnosis_mentions)):
            for j in range(i + 1, len(diagnosis_mentions)):
                if len(conflicts) >= 5:
                    break
                a = diagnosis_mentions[i]
                b = diagnosis_mentions[j]
                # Only flag cross-document conflicts with meaningfully different text
                if a["source"] == b["source"]:
                    continue
                # Skip if both are very short (noisy extractions)
                if len(a["diagnosis"]) < 8 or len(b["diagnosis"]) < 8:
                    continue
                sim = similarity(a["diagnosis"], b["diagnosis"])
                if sim < 0.4:
                    pair_key = tuple(sorted([a["diagnosis"][:40], b["diagnosis"][:40]]))
                    if pair_key in seen_conflict_pairs:
                        continue
                    seen_conflict_pairs.add(pair_key)
                    conflicts.append({
                        "type": "DIAGNOSIS_CONFLICT",
                        "severity": "HIGH",
                        "description": f"Conflicting diagnoses found between documents",
                        "source_a": a["source"],
                        "value_a": a["diagnosis"],
                        "source_b": b["source"],
                        "value_b": b["diagnosis"],
                        "requires_clinician_review": True,
                        "flagged_field": "principal_diagnosis"
                    })

        # ── 2. Medication dose conflicts ──────────────────────────────────────
        med_by_name = {}
        for section_name, pages in sections.items():
            for page in pages:
                text = page["text"]
                source = f"{page['source']} p{page['page']}"
                # Find drug + dose pairs
                for m in re.finditer(
                    r"([A-Za-z][a-z]+(?:\s[A-Za-z]+)?)\s+(\d+\.?\d*\s*(?:mg|mcg|units?|ml|g|meq))",
                    text, re.IGNORECASE
                ):
                    drug = m.group(1).strip().lower()
                    dose = m.group(2).strip().lower()
                    if len(drug) < 4:
                        continue
                    if drug not in med_by_name:
                        med_by_name[drug] = []
                    med_by_name[drug].append({"source": source, "dose": dose})

        for drug, entries in med_by_name.items():
            doses = [e["dose"] for e in entries]
            unique_doses = list(set(doses))
            if len(unique_doses) > 1:
                # Multiple different doses for same drug
                conflicts.append({
                    "type": "MEDICATION_DOSE_CONFLICT",
                    "severity": "HIGH",
                    "description": f"Inconsistent doses found for {drug}",
                    "drug": drug,
                    "doses_found": [
                        {"source": e["source"], "dose": e["dose"]} for e in entries
                    ],
                    "requires_clinician_review": True,
                    "flagged_field": "medications"
                })

        # ── 3. Date conflicts ─────────────────────────────────────────────────
        date_contexts = {}
        date_pattern = re.compile(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})')

        for section_name, pages in sections.items():
            for page in pages:
                text = page["text"]
                source = f"{page['source']} p{page['page']}"

                for ctx_pat in [
                    r"(?:admission|admit)\s*date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
                    r"discharge\s*date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
                ]:
                    for m in re.finditer(ctx_pat, text, re.IGNORECASE):
                        label = "admission_date" if "admit" in m.group(0).lower() else "discharge_date"
                        date_val = m.group(1)
                        if label not in date_contexts:
                            date_contexts[label] = []
                        date_contexts[label].append({"source": source, "date": date_val})

        for label, entries in date_contexts.items():
            unique_dates = list(set(e["date"] for e in entries))
            if len(unique_dates) > 1:
                conflicts.append({
                    "type": "DATE_CONFLICT",
                    "severity": "MEDIUM",
                    "description": f"Inconsistent {label.replace('_', ' ')} across documents",
                    "field": label,
                    "dates_found": entries,
                    "requires_clinician_review": True,
                    "flagged_field": label
                })

        # ── 4. Allergy conflicts ──────────────────────────────────────────────
        nkda_mentioned = False
        allergy_mentioned = False
        for section_name, pages in sections.items():
            for page in pages:
                text = page["text"]
                if re.search(r"\bnkda\b|no\s*known\s*(?:drug\s*)?allerg", text, re.IGNORECASE):
                    nkda_mentioned = True
                if re.search(r"allerg(?:ic|y)\s+to\s+[A-Za-z]", text, re.IGNORECASE):
                    allergy_mentioned = True

        if nkda_mentioned and allergy_mentioned:
            conflicts.append({
                "type": "ALLERGY_CONFLICT",
                "severity": "HIGH",
                "description": "Document states NKDA but also mentions a specific drug allergy",
                "requires_clinician_review": True,
                "flagged_field": "allergies"
            })

        return {"conflicts": conflicts}
