# Discharge Summary Agent

An agentic AI system that reads messy patient source documents and produces structured, clinically safe discharge summary drafts for clinician review.

The system uses a real **Plan → Act → Observe → Re-plan** agent loop rather than a single summarisation call. It dynamically decides which tools to use, identifies missing or conflicting information, performs medication reconciliation, and flags all issues that require clinician review before the summary is used.

---

## Features

### Agentic Workflow
- Plan → Act → Observe → Re-plan loop with dynamic tool selection
- Hard iteration limit (25 steps) to prevent infinite loops
- Retry logic with exponential backoff on tool failures
- Detailed trace logging: `reasoning → action → inputs → result → next_decision`

### Clinical Safety
- Strict no-hallucination policy enforced in code, not just prompts
- Missing information flagged as `[MISSING — REQUIRES CLINICIAN INPUT]`
- Pending results flagged as `[PENDING — DO NOT ASSUME VALUE]`
- Conflicting data flagged as `[CONFLICTING DATA — REQUIRES CLINICIAN RESOLUTION]`
- All outputs are drafts and require clinician review before use

### Medication Safety
- Admission vs discharge medication reconciliation
- Detects new, discontinued, and modified medications
- Flags undocumented medication changes
- Drug interaction checking (mocked API, pluggable)

### Learning System
- Simulated doctor review loop produces `(draft, corrected)` pairs
- Correction memory injects few-shot examples into future planner prompts
- Reward-based evaluation tracks improvement over rounds
- Reduced clinician edit burden over time without weakening safety guardrails

### Web UI
- Upload patient PDFs with OCR text extraction
- OCR Extracted Text Preview
- Safety flag dashboard
- Trace visualisation panel
- Learning metrics display
- OCR preview panel

---

## Architecture

```
Plan
 ↓
Act
 ↓
Observe
 ↓
Re-plan
```

### Typical execution path (8–12 steps)

```
Step 1: ingest_pdfs             → extracts all text, parses demographics/meds/etc.
Step 2: detect_conflicts        → finds cross-document contradictions
Step 3: escalate                → one flag per conflict cluster
Step 4: reconcile_meds          → compares admission vs discharge medications
Step 5: check_drug_interactions → checks discharge meds against interaction DB
Step 6: escalate                → one flag per MAJOR/MODERATE interaction found
Step 7: check_pending           → identifies pending labs/results
Step 8: escalate (×N)           → one flag per undocumented med change
Step N: build_summary           → assembles final draft
Step N+1: DONE
```

The LLM planner can deviate from this path — if ingestion fails it will retry; if no conflicts are found it skips escalation.

---

## Project Structure

```
discharge-agent/
├── main.py                     # CLI entry point
├── app.py                      # Flask web UI
├── requirements.txt
├── README.md
├── agent/
│   ├── agent_loop.py           # Core plan → execute → re-plan loop
│   └── learning_loop.py        # Learning from doctor edits
├── tools/
│   ├── pdf_ingestion.py        # PDF/text extraction + OCR fallback
│   ├── conflict_detector.py    # Cross-document contradiction detection
│   ├── med_reconciler.py       # Admission vs discharge med comparison
│   ├── drug_interaction.py     # Drug-drug interaction checker (mocked API)
│   ├── pending_checker.py      # Pending/outstanding result finder
│   ├── escalation.py           # Clinician-review flag creator
│   └── summary_builder.py      # Final draft assembler
├── data/
│   └── generate_patients.py    # Synthetic patient data generator
├── uploads/
└── output/
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run CLI (demo mode — no API key required)

```bash
# All patients, demo planner
python main.py --demo --output output

# Single patient
python main.py --patient PT-001

# Full run with Part 2 learning loop
python main.py --demo --part2 --output output
```

### Run with Anthropic API (real LLM planner)

```bash
export ANTHROPIC_API_KEY=your_key_here
python main.py --all --output output
```

### Run the web UI

```bash
python app.py
```

Then open `http://localhost:5000`

---

## No-Fabrication Guardrail

This is the highest-priority safety requirement. It is enforced in code in three places and cannot be overridden by the LLM planner:

1. **`summary_builder.py`** — every required field that cannot be found in source documents is replaced with the literal string `[MISSING — REQUIRES CLINICIAN INPUT]`. Pending results get `[PENDING — DO NOT ASSUME VALUE]`. This is hard-coded, not prompted.

2. **`pending_checker.py`** — any result marked pending is recorded verbatim and passed to the summary as-is. The tool never fills in a plausible value.

3. **`_meta.warning`** — every output contains a machine-readable warning confirming the document is a draft for review, with `auto_finalized: false`.

The LLM planner only decides *which tools to call and when*. It never directly writes clinical content into the output.

---

## Conflict and Failure Handling

**Conflicts** — `detect_conflicts` finds disagreements between documents (different diagnoses, different dates, allergy contradictions). The agent escalates them as flags. `summary_builder` marks conflicting fields with `[CONFLICTING DATA — REQUIRES CLINICIAN RESOLUTION]` — it never picks a winner.

**Tool failures** — wrapped in `try/except` with `RETRY_LIMIT = 3` and exponential backoff. Failed actions are recorded in `state.failed_actions` so the planner knows to re-plan around them. The agent never behaves as if a failed call succeeded.

**Planner failures** — if the LLM returns malformed JSON or the call fails, the agent adds an ABORT trace step and terminates cleanly.

**Step cap** — hard cap at 25 steps. If hit, a `SYSTEM_WARNING` flag is added so clinicians know the summary may be incomplete.

---

## Synthetic Test Patients

Three synthetic patients are included to test: diagnosis conflicts, medication reconciliation, drug interactions, missing information, pending results, and escalation workflows. No real patient data is used.

---

## Part 2: Learning from Doctor Edits

### Reward signal

```
reward = 1.0 - normalized_edit_distance(draft_section, corrected_section)
```

Computed per section; aggregated as mean. Range [0, 1]. Higher = less editing needed.

### Simulated reviewer

`SimulatedReviewer` applies a consistent, deterministic editing policy:
- Expands terse discharge conditions to full clinical descriptions
- Adds standard patient education and reconciliation attestation to follow-up
- Resolves diagnosis conflicts using a configurable hint (simulates clinician judgment)
- Confirms pending items are flagged and adds urgency notes

### Learning mechanism: Correction Memory

A structured key-value store of `(field, original_draft, clinician_correction)` triples. Retrieved by string similarity to the current draft and injected as few-shot examples into the planner's system prompt for relevant fields.

### Results (3 patients × 3 epochs)

| Metric | Round 1 | Round 6 | Change |
|---|---|---|---|
| Aggregate reward | 0.63 | 0.78 | +0.15 |
| Edit burden | 0.37 | 0.22 | −0.15 |
| Holdout reward | — | 0.79 | — |

### Known Limitations

**Cold start** — empty memory gives no uplift. Mitigation: seed with manually authored gold-standard examples before first deployment.

**Reward gaming** — an agent that always outputs MISSING everywhere reduces edit distance by being maximally vague. Mitigation: missing fields are penalised in reward computation; a fully-MISSING summary scores ~0.

**Style vs accuracy** — the reviewer edits style as well as medicine. Mitigation in production: separate style-edit and clinical-edit pairs; weight clinical edits 3×.

**Safety preservation** — learning only affects prompt injection for style fields (follow-up wording, condition expansion). The no-fabrication guardrail is hard-coded in `summary_builder.py`, not in any learnable prompt component.

**Simulated vs real reviewers** — real clinicians vary, disagree, and make their own errors. Production use requires IRB approval, annotation guidelines, and inter-rater reliability checks.

---

## Future Improvements

- Embedding-based retrieval for correction memory (vs string similarity)
- Section-level confidence scores propagated through to the UI
- Structured conflict resolution UI so clinicians can resolve flags inline
- ICD-10 code validation for diagnoses
- Allergy cross-check against prescribed medications
- DPO fine-tuning on (draft, corrected) pairs once N > 100
- Real PDF discharge summary export via reportlab

---

## Key Principle

The agent assists clinicians by generating safe discharge summary drafts. It can extract, compare, flag, and summarise information — but it never replaces clinician judgment. All final decisions remain with the clinician.
