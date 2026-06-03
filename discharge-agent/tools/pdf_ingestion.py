"""
PDF Ingestion Tool
==================
Reads all patient PDFs, extracts text (with OCR fallback for scanned pages),
and returns structured sections.

Never fills in missing data — missing fields are explicitly marked.
"""

import os
import re
import json
import tempfile
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

# OCR support for scanned pages
try:
    import pytesseract
    from PIL import Image
    import pypdfium2 as pdfium
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


SECTION_PATTERNS = {
    "demographics": [
        r"patient\s*name", r"date\s*of\s*birth", r"dob", r"mrn", r"medical\s*record",
        r"age", r"gender", r"sex", r"address", r"insurance"
    ],
    "admission": [
        r"admission\s*(date|note|diagnosis)", r"admitting\s*diagnosis",
        r"chief\s*complaint", r"presenting\s*complaint", r"reason\s*for\s*admission",
        r"history\s*of\s*present\s*illness", r"hpi"
    ],
    "progress_notes": [
        r"progress\s*note", r"daily\s*note", r"soap\s*note",
        r"assessment\s*and\s*plan", r"a/p", r"a\s*&\s*p"
    ],
    "discharge": [
        r"discharge\s*(summary|note|diagnosis|date|condition)",
        r"principal\s*diagnosis", r"secondary\s*diagnos",
        r"discharge\s*instructions"
    ],
    "labs": [
        r"laboratory", r"lab\s*results?", r"cbc", r"bmp", r"cmp",
        r"glucose", r"creatinine", r"sodium", r"potassium", r"wbc",
        r"hemoglobin", r"hgb", r"hematocrit", r"hct", r"platelet",
        r"pending", r"resulted", r"reference\s*range"
    ],
    "medications": [
        r"medication", r"med(s)?\b", r"drug", r"prescription",
        r"admission\s*med", r"discharge\s*med", r"current\s*med",
        r"dose", r"dosage", r"sig", r"route", r"frequency",
        r"mg\b", r"mcg\b", r"units?\b"
    ],
    "allergies": [
        r"allerg", r"nkda", r"no\s*known\s*(drug\s*)?allerg",
        r"adverse\s*reaction", r"intolerance"
    ],
    "vitals": [
        r"vital\s*signs?", r"blood\s*pressure", r"bp\b", r"heart\s*rate",
        r"hr\b", r"temperature", r"temp\b", r"resp(iratory)?\s*rate",
        r"oxygen\s*sat", r"spo2", r"o2\s*sat", r"weight", r"height", r"bmi"
    ],
    "procedures": [
        r"procedure", r"surgery", r"operation", r"surgical",
        r"imaging", r"x.?ray", r"ct\s+scan", r"mri", r"ultrasound",
        r"echocardiogram", r"echo\b", r"ekg", r"ecg"
    ],
    "follow_up": [
        r"follow.?up", r"follow up", r"f/u", r"return\s*(to|visit)",
        r"appointment", r"referral", r"outpatient"
    ],
}


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a document (PDF or plain text).
    Returns list of {page_num, text, method} dicts.
    Uses pdfplumber first; falls back to OCR for scanned pages.
    Plain text files (.txt) are read directly.
    """
    pages = []
    ext = Path(pdf_path).suffix.lower()

    # Plain text fallback — treat whole file as one page
    if ext in (".txt", ".text"):
        try:
            text = Path(pdf_path).read_text(encoding="utf-8", errors="replace")
            pages.append({"page_num": 1, "text": text.strip(), "method": "plaintext"})
            return pages
        except Exception as e:
            pages.append({"page_num": 0, "text": "", "method": "failed", "error": str(e)})
            return pages

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                # If text is too short, this is likely a scanned page
                if len(text.strip()) < 50 and HAS_OCR:
                    text = ocr_page(pdf_path, i)
                    method = "ocr"
                else:
                    method = "pdfplumber"

                pages.append({
                    "page_num": i + 1,
                    "text": text.strip(),
                    "method": method
                })
    except Exception as e:
        # Try PyPDF as fallback
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 50 and HAS_OCR:
                    text = ocr_page(pdf_path, i)
                    method = "ocr_fallback"
                else:
                    method = "pypdf_fallback"
                pages.append({
                    "page_num": i + 1,
                    "text": text.strip(),
                    "method": method
                })
        except Exception as e2:
            # Final fallback: try reading as plain text
            try:
                text = Path(pdf_path).read_text(encoding="utf-8", errors="replace")
                pages.append({"page_num": 1, "text": text.strip(), "method": "plaintext_fallback"})
            except Exception:
                pages.append({
                    "page_num": 0,
                    "text": "",
                    "method": "failed",
                    "error": str(e2)
                })

    return pages


def ocr_page(pdf_path: str, page_index: int) -> str:
    """Rasterize a PDF page and run Tesseract OCR on it."""
    if not HAS_OCR:
        return ""
    try:
        doc = pdfium.PdfDocument(pdf_path)
        page = doc[page_index]
        bitmap = page.render(scale=2.0)   # 2x = ~150 DPI equivalent
        pil_image = bitmap.to_pil()
        text = pytesseract.image_to_string(pil_image, config="--psm 6")
        return text
    except Exception:
        return ""


def classify_page(text: str) -> list[str]:
    """Identify which section types are present on a page."""
    text_lower = text.lower()
    found = []
    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                found.append(section)
                break
    return found or ["unknown"]


def extract_demographics(text: str) -> dict:
    """Parse patient demographics from text."""
    result = {}

    patterns = {
        "name": [
            r"patient\s*name[:\s]+([A-Za-z ,\-]+)",
            r"name[:\s]+([A-Za-z ,\-]{3,40})",
        ],
        "dob": [
            r"(?:dob|date\s*of\s*birth)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            r"(?:dob|date\s*of\s*birth)[:\s]+([A-Za-z]+ \d{1,2},? \d{4})",
        ],
        "mrn": [
            r"(?:mrn|medical\s*record\s*(?:number|no\.?|#?))[:\s#]*([A-Za-z0-9\-]+)",
        ],
        "age": [
            r"(?:age)[:\s]+(\d{1,3})\s*(?:y/?o|years?)",
            r"(\d{1,3})\s*(?:y/?o|year\s*old)",
        ],
        "gender": [
            r"(?:gender|sex)[:\s]+(male|female|m\b|f\b|non.binary|transgender)",
        ],
        "admission_date": [
            r"(?:admission|admit(?:ted)?)\s*date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            r"admitted\s+(?:on\s+)?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        ],
        "discharge_date": [
            r"discharge\s*date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            r"discharged\s+(?:on\s+)?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        ],
    }

    for field, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result[field] = m.group(1).strip()
                break

    return result


def extract_diagnoses(text: str) -> dict:
    """Extract principal and secondary diagnoses."""
    result = {"principal": None, "secondary": [], "discharge_condition": None}

    # Principal diagnosis — single line only
    for pat in [
        r"principal\s*diagnosis[:\s]+([^\n]{5,100})",
        r"primary\s*diagnosis[:\s]+([^\n]{5,100})",
        r"admitting\s*diagnosis[:\s]+([^\n]{5,100})",
        r"discharge\s*diagnosis[:\s]+([^\n]{5,100})",
        r"final\s*diagnosis[:\s]+([^\n]{5,100})",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["principal"] = m.group(1).strip()
            break

    # Secondary diagnoses
    sec_match = re.search(
        r"secondary\s*diagnos(?:es|is)[:\s]+((?:[^\n]+\n?){1,8})",
        text, re.IGNORECASE
    )
    if sec_match:
        lines = [l.strip().lstrip("-").strip() for l in sec_match.group(1).split("\n") if l.strip()]
        result["secondary"] = [l for l in lines if len(l) > 3][:8]

    # Discharge condition — single line, reject blank/header values
    for pat in [
        r"discharge\s*condition[:\s]+([^\n]{3,80})",
        r"condition\s*at\s*discharge[:\s]+([^\n]{3,80})",
        r"condition\s*on\s*discharge[:\s]+([^\n]{3,80})",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # Reject blank, all-caps headers, or values that look like the next field name
            if val and len(val) > 2 and not re.match(r'^[A-Z ]{3,}:?\s*$', val):
                result["discharge_condition"] = val
                break

    return result


def extract_medications(text: str) -> list[dict]:
    """
    Extract medication entries from text.
    Parses line-by-line: each line is one medication entry.
    Format: Drug Name  Dose  Route  Frequency
    """
    meds = []
    seen = set()

    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        # Skip section headers / noise lines
        if re.match(r'^[A-Z\s\(\):=\-]{10,}$', line):
            continue
        if re.match(r'^(patient|mrn|period|note|date|name|admission|discharge)\b', line, re.IGNORECASE):
            continue

        # Match: DrugName  Dose  [Route]  [Frequency]
        m = re.match(
            r'^([A-Za-z][A-Za-z0-9\s\-]+?)\s{2,}'           # drug name (2+ spaces as delimiter)
            r'(\d+\.?\d*\s*(?:mg|mcg|units?|ml|g|meq|iu))',  # dose
            line, re.IGNORECASE
        )
        if not m:
            # Fallback: single-space separated
            m = re.match(
                r'^([A-Za-z][A-Za-z0-9\s\-]{2,25}?)\s+'
                r'(\d+\.?\d*\s*(?:mg|mcg|units?|ml|g|meq|iu))',
                line, re.IGNORECASE
            )
        if m:
            name = m.group(1).strip()
            dose = m.group(2).strip()
            if len(name) < 3 or name.lower() in {"the", "and", "for", "with", "per", "po", "iv"}:
                continue
            key = (name.lower(), dose.lower())
            if key not in seen:
                seen.add(key)
                meds.append({"name": name, "dose": dose, "raw_line": line})

    return meds


def split_admission_discharge_meds(text: str) -> tuple[list[dict], list[dict]]:
    """
    Split a medication record into admission and discharge med lists
    based on section headers in the text.
    """
    admission_meds = []
    discharge_meds = []

    # Find section boundaries
    adm_match = re.search(r'admission\s+med', text, re.IGNORECASE)
    dc_match = re.search(r'discharge\s+med', text, re.IGNORECASE)

    if adm_match and dc_match:
        adm_start = adm_match.start()
        dc_start = dc_match.start()
        if adm_start < dc_start:
            adm_text = text[adm_start:dc_start]
            dc_text = text[dc_start:]
        else:
            dc_text = text[dc_start:adm_start]
            adm_text = text[adm_start:]
        admission_meds = extract_medications(adm_text)
        discharge_meds = extract_medications(dc_text)
    elif adm_match:
        admission_meds = extract_medications(text[adm_match.start():])
    elif dc_match:
        discharge_meds = extract_medications(text[dc_match.start():])
    else:
        # No clear header — treat whole block as discharge meds
        discharge_meds = extract_medications(text)

    return admission_meds, discharge_meds


def extract_allergies(text: str) -> list[str]:
    """Extract allergy information. Returns empty list if field is blank."""
    # NKDA
    if re.search(r"\bnkda\b|no\s*known\s*(?:drug\s*)?allerg", text, re.IGNORECASE):
        return ["NKDA (No Known Drug Allergies)"]

    # Find ALLERGIES: line and read only that line's value
    m = re.search(r"^allergies?\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        value = m.group(1).strip()
        # If blank or just whitespace, return empty (will become MISSING)
        if not value or len(value) < 2:
            return []
        # If it looks like a section header bleed-through, reject
        if re.match(r'^[A-Z ]+:$', value) or value.upper() == value:
            return []
        return [value]

    return []


def extract_pending_labs(text: str) -> list[dict]:
    """Find pending / not-yet-resulted lab tests. Returns clean deduplicated list."""
    pending = []
    seen = set()

    # Find lines that explicitly name a pending test
    for line in text.split("\n"):
        line_stripped = line.strip().lstrip("-•").strip()
        if not line_stripped or len(line_stripped) < 8:
            continue
        # Must contain a pending keyword
        if not re.search(r'\bpending\b|\bawaiting\b|\bnot\s+resulted\b', line_stripped, re.IGNORECASE):
            continue
        # Skip pure section headers
        if re.match(r'^(pending\s+results?|pending\s*:)\s*$', line_stripped, re.IGNORECASE):
            continue
        # Skip lines that are just "(PENDING)" or similar noise
        if re.match(r'^\(?pending\)?\.?\s*$', line_stripped, re.IGNORECASE):
            continue
        # Skip medication lines that accidentally matched
        if re.search(r'\b(mg|mcg|units?)\b', line_stripped, re.IGNORECASE):
            continue
        # Normalise: strip trailing "(PENDING)" duplicates
        clean = re.sub(r'\s*\(PENDING\)\s*$', '', line_stripped, flags=re.IGNORECASE).strip()
        clean = re.sub(r'\s*—\s*result\s+pending\s*$', '', clean, flags=re.IGNORECASE).strip()
        # Deduplicate by lowercased first 40 chars
        key = clean.lower()[:40]
        if key not in seen and len(clean) > 5:
            seen.add(key)
            pending.append({
                "test": clean,
                "status": "PENDING",
                "requires_follow_up": True,
                "note": "Result pending at time of discharge — must NOT be assumed or filled in."
            })

    return pending


def extract_follow_up(text: str) -> list[str]:
    """Extract follow-up instructions."""
    instructions = []
    m = re.search(
        r"follow.?up[:\s]+((?:[^\n]+\n?){1,15})",
        text, re.IGNORECASE
    )
    if m:
        lines = [l.strip() for l in m.group(1).split("\n") if l.strip()]
        instructions = lines[:15]
    return instructions


class PDFIngestionTool:
    """Reads all patient PDFs and returns structured clinical data."""

    def run(self, inputs: dict, state) -> dict:
        pdf_paths = inputs.get("pdf_paths", state.pdf_paths)

        all_text_by_section = {}
        all_page_results = []
        errors = []

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                errors.append(f"File not found: {pdf_path}")
                continue

            pages = extract_text_from_pdf(pdf_path)
            doc_name = Path(pdf_path).stem

            for page in pages:
                text = page["text"]
                if not text:
                    continue

                sections = classify_page(text)
                for section in sections:
                    if section not in all_text_by_section:
                        all_text_by_section[section] = []
                    all_text_by_section[section].append({
                        "source": doc_name,
                        "page": page["page_num"],
                        "text": text,
                        "method": page["method"]
                    })

                all_page_results.append({
                    "source": doc_name,
                    "page": page["page_num"],
                    "sections": sections,
                    "text_length": len(text),
                    "method": page["method"]
                })

        # Combine all text for parsing
        combined_text = "\n\n".join(
            "\n".join(p["text"] for p in pages)
            for pages in all_text_by_section.values()
        )

        # Parse structured data
        demographics = extract_demographics(combined_text)
        diagnoses = extract_diagnoses(combined_text)

        # Extract medications per document context (admission vs discharge)
        admission_meds = []
        discharge_meds = []
        # Build a fast lookup: (source, page) → text  (deduplicated)
        page_text_lookup: dict[tuple, str] = {}
        for section_pages in all_text_by_section.values():
            for sp in section_pages:
                key = (sp["source"], sp["page"])
                if key not in page_text_lookup:
                    page_text_lookup[key] = sp["text"]

        # ── Medication extraction ─────────────────────────────────────────────
        # Priority: use dedicated medication_record files first (most reliable)
        # Fallback: parse from other documents if no med record found
        med_record_pages = [
            (info["source"], info["page"])
            for info in all_page_results
            if "medication" in info["source"].lower() or "med_record" in info["source"].lower()
        ]

        if med_record_pages:
            for source_name, page_num in med_record_pages:
                page_text = page_text_lookup.get((source_name, page_num), "")
                if page_text.strip():
                    adm, dc = split_admission_discharge_meds(page_text)
                    admission_meds.extend(adm)
                    discharge_meds.extend(dc)
        else:
            for page_info in all_page_results:
                source_lower = page_info["source"].lower()
                page_text = page_text_lookup.get((page_info["source"], page_info["page"]), "")
                if not page_text.strip():
                    continue
                has_adm_header = bool(re.search(r'admission\s+med', page_text, re.IGNORECASE))
                has_dc_header  = bool(re.search(r'discharge\s+med', page_text, re.IGNORECASE))
                if has_adm_header or has_dc_header:
                    adm, dc = split_admission_discharge_meds(page_text)
                    admission_meds.extend(adm)
                    discharge_meds.extend(dc)
                elif any(k in source_lower for k in ["admission", "admit", "initial"]):
                    admission_meds.extend(extract_medications(page_text))
                elif any(k in source_lower for k in ["discharge", "dc_", "disch"]):
                    discharge_meds.extend(extract_medications(page_text))

        allergies = extract_allergies(combined_text)
        pending_labs = extract_pending_labs(combined_text)
        follow_up = extract_follow_up(combined_text)

        # Hospital course from progress notes
        hospital_course_texts = []
        for page in all_text_by_section.get("progress_notes", []):
            hospital_course_texts.append(
                f"[{page['source']} p{page['page']}]: {page['text'][:500]}"
            )
        for page in all_text_by_section.get("discharge", []):
            hospital_course_texts.append(
                f"[{page['source']} p{page['page']}]: {page['text'][:500]}"
            )

        return {
            "sections": all_text_by_section,
            "page_inventory": all_page_results,
            "errors": errors,
            "structured": {
                "demographics": demographics,
                "diagnoses": diagnoses,
                "admission_medications": admission_meds,
                "discharge_medications": discharge_meds,
                "allergies": allergies,
                "pending_labs": pending_labs,
                "follow_up_instructions": follow_up,
                "hospital_course_raw": hospital_course_texts,
            }
        }
