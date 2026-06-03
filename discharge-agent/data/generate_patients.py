"""
Synthetic Patient Data Generator
==================================
Creates realistic, intentionally messy hospital documents for testing.
Each patient has multiple PDFs simulating the kinds of issues the agent must handle:
- Missing fields
- Pending labs
- Medication changes without documented reasons
- Conflicting diagnoses between notes
- OCR noise in some documents

All data is entirely synthetic. No real patient data.
"""

import os
import json
import random
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ── Patient templates ──────────────────────────────────────────────────────────

PATIENTS = [
    {
        "id": "PT-001",
        "name": "Margaret Chen",
        "dob": "03/14/1958",
        "age": 66,
        "gender": "Female",
        "mrn": "MRN-884721",
        "admission_date": "05/12/2024",
        "discharge_date": "05/17/2024",
        "admission_dx": "Diabetic Ketoacidosis (DKA)",
        "discharge_dx_note1": "Type 2 Diabetes Mellitus with Diabetic Ketoacidosis",  # admission note says this
        "discharge_dx_note2": "Urinary Tract Infection",  # progress note CONFLICTS
        "secondary_dx": ["Hypertension", "Chronic Kidney Disease Stage 3", "Obesity"],
        "discharge_condition": "Stable",
        "admission_meds": [
            ("Metformin", "1000 mg", "PO", "BID"),
            ("Lisinopril", "10 mg", "PO", "Daily"),
            ("Atorvastatin", "40 mg", "PO", "Nightly"),
            ("Aspirin", "81 mg", "PO", "Daily"),
        ],
        "discharge_meds": [
            ("Insulin Glargine", "20 units", "SC", "Nightly"),   # NEW — no reason documented
            ("Lisinopril", "20 mg", "PO", "Daily"),               # DOSE CHANGE — no reason
            ("Atorvastatin", "40 mg", "PO", "Nightly"),
            ("Aspirin", "81 mg", "PO", "Daily"),
            # Metformin DISCONTINUED — no reason documented
        ],
        "allergies": "Penicillin (hives)",
        "pending_labs": ["Urine culture (sent 05/16) — result pending", "HbA1c — pending (sent to reference lab)"],
        "follow_up": [
            "Follow up with primary care physician in 1 week",
            "Endocrinology referral — appointment to be scheduled",
            "Recheck BMP in 1 week",
        ],
        "hospital_course": (
            "66F with known T2DM presenting with DKA. pH 7.21 on admission. "
            "Started on insulin drip; transitioned to subcutaneous insulin after anion gap closure. "
            "Received 3L IV NS. Potassium repleted. Blood glucose normalized. "
            "Nephrology consulted for CKD monitoring."
        ),
        "procedures": ["IV access x2", "Continuous cardiac monitoring", "Urinalysis", "Chest X-ray (unremarkable)"],
        "scenario_notes": "DKA patient with diagnosis conflict (admission says DKA, progress note says UTI), Metformin stopped without reason, Lisinopril dose doubled without reason, pending urine culture and HbA1c.",
    },
    {
        "id": "PT-002",
        "name": "Robert Okafor",
        "dob": "11/28/1945",
        "age": 78,
        "gender": "Male",
        "mrn": "MRN-334556",
        "admission_date": "05/09/2024",
        "discharge_date": "05/14/2024",
        "admission_dx": "Acute Exacerbation of COPD",
        "discharge_dx_note1": "Acute Exacerbation of COPD, Community-Acquired Pneumonia",
        "discharge_dx_note2": "Acute Exacerbation of COPD, Community-Acquired Pneumonia",
        "secondary_dx": ["Atrial Fibrillation", "Heart Failure with reduced EF (HFrEF)", "Type 2 Diabetes"],
        "discharge_condition": "Improved",
        "admission_meds": [
            ("Warfarin", "5 mg", "PO", "Daily"),
            ("Digoxin", "0.125 mg", "PO", "Daily"),
            ("Metoprolol Succinate", "50 mg", "PO", "Daily"),
            ("Furosemide", "40 mg", "PO", "Daily"),
            ("Tiotropium", "18 mcg", "Inhaled", "Daily"),
            ("Albuterol", "2.5 mg", "Neb", "Q4H PRN"),
        ],
        "discharge_meds": [
            ("Warfarin", "5 mg", "PO", "Daily"),
            ("Amiodarone", "200 mg", "PO", "Daily"),            # NEW — interacts with Warfarin AND Digoxin
            ("Digoxin", "0.125 mg", "PO", "Daily"),             # DANGEROUS: Amiodarone raises Digoxin levels
            ("Metoprolol Succinate", "50 mg", "PO", "Daily"),
            ("Furosemide", "40 mg", "PO", "Daily"),
            ("Tiotropium", "18 mcg", "Inhaled", "Daily"),
            ("Albuterol", "2.5 mg", "Neb", "Q4H PRN"),
            ("Azithromycin", "250 mg", "PO", "Daily x5 days"),   # NEW for CAP — reason in notes
            ("Prednisone", "40 mg", "PO", "Daily x5 days"),      # NEW for COPD exac — reason in notes
        ],
        "allergies": "NKDA",
        "pending_labs": [
            "Sputum culture — pending (ordered 05/09)",
            "INR (draw 05/15) — pending, warfarin dose adjustment required once available",
        ],
        "follow_up": [
            "Pulmonology follow-up in 2 weeks",
            "Cardiology follow-up in 1 week — INR check and amiodarone monitoring",
            "Repeat CXR in 4 weeks",
            "Resume Warfarin per home dose; INR to be checked by PCP within 3 days",
        ],
        "hospital_course": (
            "78M with COPD presenting with 3 days worsening dyspnea and productive cough. "
            "CXR showed RLL infiltrate consistent with CAP. "
            "Started on IV steroids, bronchodilators, and azithromycin. "
            "Cardiology consulted for new AF with RVR — amiodarone started for rate control. "
            "INR supratherapeutic at 3.8 on admission; warfarin held x2 days then restarted. "
            "Improved clinically; O2 requirement decreased."
        ),
        "procedures": [
            "Chest X-ray (RLL infiltrate)",
            "CT chest without contrast",
            "Sputum collection for culture",
            "Echocardiogram (EF 35%)",
        ],
        "scenario_notes": "COPD/CAP patient. Major drug interactions: Amiodarone + Warfarin (MAJOR), Amiodarone + Digoxin (MAJOR). Pending sputum culture and INR.",
    },
    {
        "id": "PT-003",
        "name": "Sofia Ramirez",
        "dob": "07/04/1989",
        "age": 34,
        "gender": "Female",
        "mrn": "MRN-671209",
        "admission_date": "05/15/2024",
        "discharge_date": "05/16/2024",
        # Intentionally minimal — many fields missing
        "admission_dx": "Acute Gastroenteritis",
        "discharge_dx_note1": "Acute Gastroenteritis vs. Early Appendicitis — see surgical note",
        "discharge_dx_note2": "Acute Gastroenteritis",
        "secondary_dx": [],
        "discharge_condition": "",  # MISSING
        "admission_meds": [],       # Not documented
        "discharge_meds": [
            ("Ondansetron", "4 mg", "PO", "Q8H PRN nausea"),
            ("Ibuprofen", "400 mg", "PO", "Q6H PRN pain"),
        ],
        "allergies": "",            # MISSING
        "pending_labs": [
            "CT abdomen/pelvis — ordered, result PENDING",
            "Beta-hCG — ordered (rule out ectopic) — PENDING",
        ],
        "follow_up": [
            "Return to ED if pain worsens, fever, or unable to tolerate PO",
            "Follow up with PCP in 48–72 hours",
        ],
        "hospital_course": (
            "34F presenting with 12h acute onset RLQ pain, nausea, vomiting. "
            "Vital signs stable. WBC 11.2. Surgical consult obtained — equivocal exam. "
            "CT abdomen/pelvis ordered but result pending at time of discharge. "
            "Tolerated PO challenge. Patient elected to leave."
        ),
        "procedures": [
            "IV access",
            "IV fluids 1L NS",
            "CT abdomen/pelvis (pending result)",
            "Surgical consultation",
        ],
        "scenario_notes": "Short stay, many missing fields (allergies, discharge condition, admission meds). Conflicting diagnosis (gastroenteritis vs. appendicitis). Critical pending results: CT abdomen and beta-hCG.",
    },
]


def generate_text_documents(patient: dict, output_dir: str):
    """Generate plain-text versions of patient documents."""
    os.makedirs(output_dir, exist_ok=True)
    files = []

    # ── 1. Admission Note ────────────────────────────────────────────────────
    admission_text = f"""
ADMISSION NOTE
==============
Patient Name: {patient['name']}
MRN: {patient['mrn']}
Date of Birth: {patient['dob']}
Age: {patient['age']} years old
Gender: {patient['gender']}

Admission Date: {patient['admission_date']}

ADMITTING DIAGNOSIS: {patient['admission_dx']}

ALLERGIES: {patient['allergies'] if patient['allergies'] else ''}

CHIEF COMPLAINT:
Patient presents with {patient['admission_dx'].lower()}.

HISTORY OF PRESENT ILLNESS:
{patient['hospital_course']}

ADMISSION MEDICATIONS:
{''.join(f"- {m[0]} {m[1]} {m[3]}" + chr(10) for m in patient['admission_meds']) if patient['admission_meds'] else "None on file"}

VITAL SIGNS ON ADMISSION:
Blood Pressure: 138/88 mmHg
Heart Rate: 96 bpm
Temperature: 37.8 C
Respiratory Rate: 18/min
O2 Saturation: 97% on room air

PHYSICAL EXAMINATION:
General: Alert and oriented x3. Mildly distressed.
Pertinent findings per admitting note.

PLAN:
{chr(10).join(f'- {p}' for p in patient['procedures'])}

Signed: Dr. [Admitting Physician]
"""
    p = Path(output_dir) / "admission_note.txt"
    p.write_text(admission_text)
    files.append(str(p))

    # ── 2. Progress Note (with intentional conflict) ─────────────────────────
    progress_text = f"""
PROGRESS NOTE — Day 2
======================
Patient: {patient['name']}    MRN: {patient['mrn']}
Date: {patient['admission_date'].replace(patient['admission_date'].split('/')[1], str(int(patient['admission_date'].split('/')[1])+1))}

SUBJECTIVE:
Patient reports improved symptoms. Tolerating medications.

OBJECTIVE:
Vitals: BP 132/80, HR 88, Temp 37.2, RR 16, O2 Sat 98% RA

ASSESSMENT AND PLAN:
Discharge Diagnosis: {patient['discharge_dx_note2']}

{chr(10).join(f'#{i+1}. {dx}' for i, dx in enumerate(patient['secondary_dx'])) if patient['secondary_dx'] else ''}

Plan:
- Continue current medications
- Monitor labs
{''.join(f"- {lab}" + chr(10) for lab in patient['pending_labs'])}

Signed: Dr. [Covering Physician]
"""
    p = Path(output_dir) / "progress_note_day2.txt"
    p.write_text(progress_text)
    files.append(str(p))

    # ── 3. Discharge Summary ─────────────────────────────────────────────────
    discharge_text = f"""
DISCHARGE SUMMARY
=================
Patient Name: {patient['name']}
MRN: {patient['mrn']}
Date of Birth: {patient['dob']}
Admission Date: {patient['admission_date']}
Discharge Date: {patient['discharge_date']}

PRINCIPAL DIAGNOSIS: {patient['discharge_dx_note1']}

SECONDARY DIAGNOSES:
{''.join(f'- {dx}' + chr(10) for dx in patient['secondary_dx']) if patient['secondary_dx'] else '- None documented'}

DISCHARGE CONDITION: {patient['discharge_condition']}

ALLERGIES: {patient['allergies']}

HOSPITAL COURSE:
{patient['hospital_course']}

PROCEDURES PERFORMED:
{''.join(f'- {p}' + chr(10) for p in patient['procedures'])}

DISCHARGE MEDICATIONS:
{''.join(f'- {m[0]} {m[1]} {m[2]} {m[3]}' + chr(10) for m in patient['discharge_meds']) if patient['discharge_meds'] else ''}

PENDING RESULTS AT DISCHARGE:
{''.join(f'- {lab} (PENDING)' + chr(10) for lab in patient['pending_labs']) if patient['pending_labs'] else '- None'}

FOLLOW-UP INSTRUCTIONS:
{''.join(f'- {f}' + chr(10) for f in patient['follow_up'])}

Signed: Dr. [Discharging Physician]
"""
    p = Path(output_dir) / "discharge_summary.txt"
    p.write_text(discharge_text)
    files.append(str(p))

    # ── 4. Lab Results ───────────────────────────────────────────────────────
    lab_text = f"""
LABORATORY RESULTS
==================
Patient: {patient['name']}    MRN: {patient['mrn']}

COMPLETE BLOOD COUNT (Admission):
WBC:        11.2  [4.5-11.0]  H
Hemoglobin: 12.8  [12.0-16.0]
Hematocrit: 38.5  [36.0-48.0]
Platelets:  245   [150-400]

COMPREHENSIVE METABOLIC PANEL:
Sodium:     138   [136-145]
Potassium:  3.2   [3.5-5.1]   L
Chloride:   101   [98-107]
CO2:        18    [22-29]     L
BUN:        24    [7-20]      H
Creatinine: 1.4   [0.6-1.2]   H
Glucose:    412   [70-100]    H
eGFR:       48    mL/min      (CKD Stage 3)

PENDING RESULTS:
{''.join(f'{lab}' + chr(10) for lab in patient['pending_labs'])}

Report generated by: Laboratory Information System
"""
    p = Path(output_dir) / "lab_results.txt"
    p.write_text(lab_text)
    files.append(str(p))

    # ── 5. Medication Record ─────────────────────────────────────────────────
    med_text = f"""
MEDICATION ADMINISTRATION RECORD
==================================
Patient: {patient['name']}    MRN: {patient['mrn']}
Period: {patient['admission_date']} to {patient['discharge_date']}

ADMISSION MEDICATIONS (Home Medications on Admission):
{''.join(f'{m[0]}  {m[1]}  {m[2]}  {m[3]}' + chr(10) for m in patient['admission_meds']) if patient['admission_meds'] else ''}

DISCHARGE MEDICATIONS:
{''.join(f'{m[0]}  {m[1]}  {m[2]}  {m[3]}' + chr(10) for m in patient['discharge_meds']) if patient['discharge_meds'] else 'NONE'}

NOTE: Changes from admission medications require clinician reconciliation.
"""
    p = Path(output_dir) / "medication_record.txt"
    p.write_text(med_text)
    files.append(str(p))

    return files


def generate_all_patients(base_dir: str = "data/patients") -> dict:
    """Generate all synthetic patient datasets."""
    os.makedirs(base_dir, exist_ok=True)
    patient_registry = {}

    for patient in PATIENTS:
        pid = patient["id"]
        patient_dir = os.path.join(base_dir, pid)
        os.makedirs(patient_dir, exist_ok=True)

        files = generate_text_documents(patient, patient_dir)
        patient_registry[pid] = {
            "patient_id": pid,
            "name": patient["name"],
            "scenario_notes": patient["scenario_notes"],
            "files": files,
        }
        print(f"Generated patient {pid}: {len(files)} documents in {patient_dir}")

    # Save registry
    registry_path = os.path.join(base_dir, "registry.json")
    with open(registry_path, "w") as f:
        json.dump(patient_registry, f, indent=2)

    print(f"\nPatient registry saved to {registry_path}")
    return patient_registry


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "data/patients"
    registry = generate_all_patients(base)
    for pid, info in registry.items():
        print(f"\n{pid}: {info['name']}")
        print(f"  Scenario: {info['scenario_notes']}")
        print(f"  Files: {info['files']}")
