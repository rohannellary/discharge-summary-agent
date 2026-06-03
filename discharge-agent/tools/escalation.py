"""
Escalation Tool
================
Creates a formal clinician-review flag for a concern.
Every safety issue, conflict, interaction, and missing required field
gets a structured escalation record.
"""

import time
from typing import Any


SEVERITY_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

ESCALATION_TYPES = {
    "MISSING_REQUIRED_FIELD",
    "DIAGNOSIS_CONFLICT",
    "MEDICATION_CONFLICT",
    "DRUG_INTERACTION",
    "UNDOCUMENTED_MED_CHANGE",
    "PENDING_CRITICAL_RESULT",
    "DATA_QUALITY",
    "SYSTEM_WARNING",
}


class EscalationTool:
    """
    Creates a structured escalation / clinician flag.
    All escalations are surfaced in the final summary — never suppressed.
    """

    def run(self, inputs: dict, state) -> dict:
        escalation_type = inputs.get("type", "DATA_QUALITY")
        severity = inputs.get("severity", "MEDIUM")
        message = inputs.get("message", "Requires clinician review")
        field = inputs.get("field")
        details = inputs.get("details", {})

        if severity not in SEVERITY_LEVELS:
            severity = "MEDIUM"

        flag = {
            "id": f"FLAG-{len(state.flags) + 1:04d}",
            "type": escalation_type,
            "severity": severity,
            "message": message,
            "field": field,
            "details": details,
            "requires_clinician_review": True,
            "auto_resolved": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "This item has been flagged for mandatory clinician review before the discharge summary can be finalized."
        }

        return flag
