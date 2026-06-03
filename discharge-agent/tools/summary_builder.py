"""
Summary Builder Tool
=====================
Assembles the final structured discharge-summary draft from all gathered state.

CORE GUARDRAIL: Every required field that cannot be sourced from documents is
explicitly marked MISSING or PENDING. Nothing is ever fabricated.
The output is always labelled a DRAFT FOR CLINICIAN REVIEW.
"""

import re
import time
from typing import Any

REQUIRED_FIELDS = [
    "patient_name", "mrn", "dob", "admission_date", "discharge_date",
    "principal_diagnosis", "hospital_course", "discharge_medications",
    "allergies", "discharge_condition", "follow_up_instructions"
]

MISSING_SENTINEL = "[MISSING — REQUIRES CLINICIAN INPUT]"
PENDING_SENTINEL = "[PENDING — DO NOT ASSUME VALUE]"


def safe_get(structured: dict, *keys, default=None):
    """Safely traverse nested dict."""
    val = structured
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
        if val is None:
            return default
    return val


class SummaryBuilderTool:
    """
    Builds the final structured discharge summary draft.
    Missing data → MISSING sentinel. Pending data → PENDING sentinel.
    Never fabricates.
    """

    def run(self, inputs: dict, state) -> dict:
        structured = state.structured_data
        demo = structured.get("demographics", {})
        diagnoses = structured.get("diagnoses", {})
        pending_labs = state.pending_items
        conflicts = state.conflicts
        flags = state.flags
        med_recon = state.med_reconciliation or {}
        drug_interactions = state.drug_interactions or []

        # ── Demographics ──────────────────────────────────────────────────────
        patient_name = demo.get("name") or MISSING_SENTINEL
        mrn = demo.get("mrn") or MISSING_SENTINEL
        dob = demo.get("dob") or MISSING_SENTINEL
        age = demo.get("age") or MISSING_SENTINEL
        gender = demo.get("gender") or MISSING_SENTINEL
        admission_date = demo.get("admission_date") or MISSING_SENTINEL
        discharge_date = demo.get("discharge_date") or MISSING_SENTINEL

        # ── Diagnoses ─────────────────────────────────────────────────────────
        principal_dx = diagnoses.get("principal")
        if conflicts:
            # If there's a diagnosis conflict, mark both and flag
            dx_conflicts = [c for c in conflicts if c.get("type") == "DIAGNOSIS_CONFLICT"]
            if dx_conflicts:
                conflict_desc = "; ".join(
                    f"[{c['source_a']}: {c['value_a']}] vs [{c['source_b']}: {c['value_b']}]"
                    for c in dx_conflicts[:2]
                )
                principal_dx = f"[CONFLICTING DATA — REQUIRES CLINICIAN RESOLUTION]: {conflict_desc}"
            elif not principal_dx:
                principal_dx = MISSING_SENTINEL
        elif not principal_dx:
            principal_dx = MISSING_SENTINEL

        secondary_dx = diagnoses.get("secondary", [])
        discharge_condition = diagnoses.get("discharge_condition") or MISSING_SENTINEL

        # ── Hospital Course ───────────────────────────────────────────────────
        course_raw = structured.get("hospital_course_raw", [])

        # Prefer discharge summary section; fall back to progress notes
        discharge_course = [c for c in course_raw if "discharge_summary" in c.lower() or "discharge" in c.lower()]
        progress_course  = [c for c in course_raw if "progress" in c.lower()]
        ordered_course   = discharge_course or progress_course or course_raw

        # Strip source/page prefixes like "[progress_note_day2 p1]: PROGRESS NOTE — Day 2\n======"
        cleaned_snippets = []
        for snippet in ordered_course[:3]:
            # Remove the [source pN]: prefix
            snippet = re.sub(r'^\[.+?\]:\s*', '', snippet).strip()
            # Remove repeated = or - header lines
            snippet = re.sub(r'^[=\-]{3,}\s*\n', '', snippet, flags=re.MULTILINE)
            # Remove document title lines (all-caps lines ≤ 40 chars)
            lines = snippet.split('\n')
            content_lines = [l for l in lines if not re.match(r'^[A-Z\s\-=]{5,40}$', l.strip())]
            snippet = '\n'.join(content_lines).strip()
            if snippet and len(snippet) > 30:
                cleaned_snippets.append(snippet)

        if cleaned_snippets:
            hospital_course = " | ".join(cleaned_snippets)[:1200]
        else:
            hospital_course = MISSING_SENTINEL

        # ── Medications ───────────────────────────────────────────────────────
        discharge_meds_raw = structured.get("discharge_medications", [])
        if discharge_meds_raw:
            discharge_medications = []
            changes = med_recon.get("changes", [])
            for med in discharge_meds_raw:
                name = med.get("name", "Unknown")
                dose = med.get("dose", "")
                # Find reconciliation status
                change = next((c for c in changes
                               if c.get("drug", "").lower() in name.lower()
                               or name.lower() in c.get("drug", "").lower()), None)
                status = ""
                flag = ""
                if change:
                    if change["type"] == "NEW":
                        status = "NEW (not on admission list)"
                        if change.get("flag_for_review"):
                            flag = "⚠ UNDOCUMENTED — reason not found"
                    elif change["type"] == "DOSE_CHANGED":
                        status = f"DOSE CHANGED from {change['admission_dose']}"
                        if change.get("flag_for_review"):
                            flag = "⚠ UNDOCUMENTED — reason not found"
                    elif change["type"] == "CONTINUED":
                        status = "Continued from admission"

                discharge_medications.append({
                    "name": name,
                    "dose": dose,
                    "reconciliation_status": status or "Unknown — verify",
                    "flag": flag
                })
        else:
            discharge_medications = [{"name": MISSING_SENTINEL, "dose": "", "reconciliation_status": "", "flag": ""}]

        # Discontinued medications
        discontinued = [
            c for c in med_recon.get("changes", []) if c.get("type") == "DISCONTINUED"
        ]

        # ── Allergies ─────────────────────────────────────────────────────────
        allergies = structured.get("allergies", [])
        # Check for allergy conflict
        allergy_conflicts = [c for c in conflicts if c.get("type") == "ALLERGY_CONFLICT"]
        if allergy_conflicts:
            allergies = [f"[CONFLICTING ALLERGY DATA — REQUIRES REVIEW]: {a}" for a in allergies]
        if not allergies:
            allergies = [MISSING_SENTINEL]

        # ── Pending Results ───────────────────────────────────────────────────
        pending_results = []
        for item in pending_labs:
            pending_results.append({
                "test": item.get("test_or_result", item.get("test", "Unknown test")),
                "status": "PENDING",
                "note": "Result not available at time of discharge — follow-up required"
            })

        # ── Follow-up ─────────────────────────────────────────────────────────
        follow_up = structured.get("follow_up_instructions", [])
        if not follow_up:
            follow_up = [MISSING_SENTINEL]

        # ── Procedures ────────────────────────────────────────────────────────
        procedures = []
        proc_sources = []

        # Collect all text from documents
        for section_name, pages in state.extracted_sections.items():
            for page in pages:
                text = page["text"]
                source = f"{page['source']} p{page['page']}"

                # Find procedures block
                for block_pat in [
                    r"procedures?\s+performed[:\s]+((?:[^\n]+\n?){1,10})",
                    r"procedures?[:\s]+((?:[-•]\s*[^\n]+\n?){1,10})",
                ]:
                    bm = re.search(block_pat, text, re.IGNORECASE)
                    if bm:
                        lines = bm.group(1).split("\n")
                        for line in lines:
                            proc = line.strip().lstrip("-•").strip()
                            if len(proc) > 5 and proc not in procedures:
                                # Filter noise: skip lines that look like section headers
                                if not re.match(r'^[A-Z\s]{10,}:?\s*$', proc):
                                    procedures.append(proc)

                # Also grab individual procedure mentions
                for pat in [
                    r"(?:procedure|surgery|operation)[:\s]+([^\n]{5,100})",
                    r"(?:imaging|ct\s+scan|mri|ultrasound|x.?ray|echo(?:cardiogram)?|ekg|ecg)[:\s]+([^\n]{5,80})",
                ]:
                    for m in re.finditer(pat, text, re.IGNORECASE):
                        proc = m.group(1).strip()
                        if proc and len(proc) > 4 and proc not in procedures:
                            if not re.match(r'^[A-Z\s]{10,}:?\s*$', proc):
                                procedures.append(proc)

        if not procedures:
            procedures = ["[MISSING — REQUIRES CLINICIAN INPUT]"]

        # ── Drug interactions summary ─────────────────────────────────────────
        interaction_summary = []
        for ix in drug_interactions:
            if ix.get("requires_clinician_review"):
                interaction_summary.append({
                    "drugs": f"{ix['drug_a']} + {ix['drug_b']}",
                    "severity": ix["severity"],
                    "description": ix["description"]
                })

        # ── All clinician flags ───────────────────────────────────────────────
        all_flags = list(flags)  # includes escalations added by agent

        # Add flags for any still-missing required fields
        missing_fields = []
        for field_name in REQUIRED_FIELDS:
            val = None
            if field_name == "patient_name": val = patient_name
            elif field_name == "mrn": val = mrn
            elif field_name == "dob": val = dob
            elif field_name == "admission_date": val = admission_date
            elif field_name == "discharge_date": val = discharge_date
            elif field_name == "principal_diagnosis": val = principal_dx
            elif field_name == "hospital_course": val = hospital_course
            elif field_name == "discharge_condition": val = discharge_condition

            if val and MISSING_SENTINEL in str(val):
                missing_fields.append(field_name)

        if missing_fields:
            all_flags.append({
                "type": "MISSING_REQUIRED_FIELD",
                "severity": "HIGH",
                "message": f"Required fields not found in source documents: {', '.join(missing_fields)}",
                "field": missing_fields,
                "requires_clinician_review": True
            })

        # ── Assemble final draft ──────────────────────────────────────────────
        summary = {
            "_meta": {
                "document_type": "DISCHARGE SUMMARY DRAFT — FOR CLINICIAN REVIEW ONLY",
                "auto_finalized": False,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "patient_id": state.patient_id,
                "warning": (
                    "THIS IS AN AI-GENERATED DRAFT. It must be reviewed and verified by a "
                    "licensed clinician before use. Fields marked MISSING require clinician "
                    "input. Fields marked PENDING await outstanding results. "
                    "Conflicts must be resolved by the treating team."
                ),
                "total_flags": len(all_flags),
                "requires_review_before_finalization": True
            },
            "patient_demographics": {
                "name": patient_name,
                "mrn": mrn,
                "date_of_birth": dob,
                "age": age,
                "gender": gender,
            },
            "admission_discharge_dates": {
                "admission_date": admission_date,
                "discharge_date": discharge_date,
            },
            "diagnoses": {
                "principal_diagnosis": principal_dx,
                "secondary_diagnoses": secondary_dx if secondary_dx else [MISSING_SENTINEL],
            },
            "hospital_course": hospital_course,
            "procedures": procedures,
            "allergies": allergies,
            "medications": {
                "discharge_medications": discharge_medications,
                "discontinued_at_discharge": [
                    {
                        "name": d["drug"],
                        "admission_dose": d["admission_dose"],
                        "reason": d.get("reason_documented") or "NOT DOCUMENTED — REQUIRES CLINICIAN REVIEW",
                        "flag": "⚠ UNDOCUMENTED" if d.get("flag_for_review") else ""
                    }
                    for d in discontinued
                ] if discontinued else [],
            },
            "pending_results": pending_results if pending_results else [
                {"test": "None identified as pending", "status": "REVIEW", "note": "Verify no pending results exist"}
            ],
            "follow_up_instructions": follow_up,
            "discharge_condition": discharge_condition,
            "drug_interactions_flagged": interaction_summary,
            "clinician_review_flags": all_flags,
            "conflicts_detected": conflicts,
        }

        return summary
