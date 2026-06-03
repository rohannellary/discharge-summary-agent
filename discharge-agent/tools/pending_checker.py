"""
Pending Checker Tool
=====================
Identifies pending / not-yet-resulted labs, imaging, and consults.
These MUST appear in the output — never filled with plausible values.
"""

import re
from typing import Any


PENDING_KEYWORDS = [
    "pending", "awaiting", "not resulted", "not finalized", "preliminary",
    "sent out", "in progress", "ordered", "requested", "ref lab",
    "send out", "final pending", "result pending", "tbd", "to be done"
]

LAB_NAMES = [
    "culture", "sensitivity", "blood culture", "urine culture", "sputum culture",
    "pcr", "bnp", "troponin", "procalcitonin", "sed rate", "esr", "crp",
    "tsh", "free t4", "t3", "cortisol", "hba1c", "a1c",
    "vitamin d", "b12", "folate", "iron studies", "ferritin",
    "hepatitis", "hep b", "hep c", "hiv", "rpr", "anca", "ana",
    "echocardiogram", "echo", "stress test", "holter", "cardiac mri",
    "ct scan", "mri", "pet scan", "ultrasound", "x-ray",
    "colonoscopy", "endoscopy", "biopsy", "pathology",
    "pulmonary function", "pft", "sleep study"
]


class PendingCheckerTool:
    """Finds all pending/outstanding results across all documents."""

    def run(self, inputs: dict, state) -> dict:
        pending = []
        sections = state.extracted_sections

        for section_name, pages in sections.items():
            for page in pages:
                text = page["text"]
                source = f"{page['source']} p{page['page']}"
                text_lower = text.lower()

                for keyword in PENDING_KEYWORDS:
                    if keyword not in text_lower:
                        continue

                    # Find lines containing the keyword
                    for line in text.split("\n"):
                        if keyword.lower() in line.lower():
                            # Extract what's pending
                            cleaned = line.strip()
                            if len(cleaned) < 5:
                                continue

                            item = {
                                "test_or_result": cleaned[:120],
                                "status": "PENDING",
                                "source": source,
                                "requires_follow_up": True,
                                "note": f"Result pending — must NOT be assumed or filled in. Requires clinician follow-up."
                            }

                            # Avoid duplicates
                            if not any(p["test_or_result"] == item["test_or_result"] for p in pending):
                                pending.append(item)

        # Also check for labs that are named but have no result value
        for section_name, pages in sections.items():
            for page in pages:
                text = page["text"]
                source = f"{page['source']} p{page['page']}"

                for lab in LAB_NAMES:
                    pattern = rf"\b{re.escape(lab)}\b[:\s]+(?:pending|tbd|\?|--|-|n/?a)"
                    for m in re.finditer(pattern, text, re.IGNORECASE):
                        item = {
                            "test_or_result": m.group(0).strip()[:120],
                            "status": "PENDING",
                            "source": source,
                            "requires_follow_up": True,
                            "note": "Lab ordered but no result present in documents."
                        }
                        if not any(p["test_or_result"] == item["test_or_result"] for p in pending):
                            pending.append(item)

        # Deduplicate: normalise each item, keep shortest clean version per test name
        def normalise_pending(text):
            t = text.strip().lstrip('-').strip()
            # Remove trailing "(PENDING)" duplicates
            t = re.sub(r'\s*\(PENDING\)\s*$', '', t, flags=re.IGNORECASE).strip()
            return t

        seen_normalised = set()
        deduped = []
        for item in pending:
            norm = normalise_pending(item['test_or_result']).lower()
            # Skip pure section headers like "PENDING RESULTS AT DISCHARGE:"
            if re.match(r'^pending\s+results', norm, re.IGNORECASE):
                continue
            if re.match(r'^pending\s*:?\s*$', norm, re.IGNORECASE):
                continue
            # Skip if we already have a version of this test
            core = re.sub(r'\s*(pending|awaiting|ordered|sent\s+\d{2}/\d{2}).*$', '', norm).strip()
            if len(core) < 5:
                continue
            if core not in seen_normalised:
                seen_normalised.add(core)
                item['test_or_result'] = normalise_pending(item['test_or_result'])
                deduped.append(item)

        pending = deduped

        return {
            "pending": pending,
            "count": len(pending),
            "note": f"Found {len(pending)} pending items. These must appear in the discharge summary as PENDING — never filled with assumed values."
        }
