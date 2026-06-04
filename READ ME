# Discharge Summary Agent

An agentic AI system that reads messy patient source documents and produces
structured, clinically safe discharge summary drafts for clinician review.

---

## Quick Start

```bash
# Demo mode — no API key required, uses rule-based planner
python main.py --demo --output output

# With Anthropic API (real LLM planner)
export ANTHROPIC_API_KEY=your_key_here
python main.py --all --output output

# Single patient
python main.py --patient PT-001

# Full run with Part 2 learning loop
python main.py --demo --part2 --output output
```

---

## Project Structure

```
discharge-agent/
├── main.py                        # Entry point
├── agent/
│   ├── agent_loop.py              # Core agentic loop (plan → execute → re-plan)
│   └── learning_loop.py           # Part 2: learning from doctor edits
├── tools/
│   ├── pdf_ingestion.py           # PDF/text extraction + OCR fallback
│   ├── conflict_detector.py       # Cross-document contradiction detection
│   ├── med_reconciler.py          # Admission vs discharge med comparison
│   ├── drug_interaction.py        # Drug-drug interaction checker (mocked API)
│   ├── pending_checker.py         # Pending/outstanding result finder
│   ├── escalation.py              # Clinician-review flag creator
│   └── summary_builder.py         # Final draft assembler
└── data/
    └── generate_patients.py       # Synthetic patient data generator
```

---

## Agent Loop Design

The agent follows a **plan → execute → observe → re-plan** loop:

1. **Planner LLM call**: Given current state (what's been extracted, what
   tools have run, what was found), the LLM reasons about what to do next
   and returns a single structured action.

2. **Tool executor**: Runs the chosen tool with retry logic (up to 3 attempts
   with backoff). On failure: records the failure, never pretends it succeeded.

3. **State updater**: Merges tool output into agent state.

4. **Hard step cap**: `MAX_STEPS = 25`. If hit, a system flag is added and the
   partial summary is output — the agent never runs forever.

5. **Trace emitter**: Every step logs `reasoning → action → inputs → result
   → next_decision` with timestamps and durations.

### Typical execution path (8–12 steps)

```
Step 1: ingest_pdfs        → extracts all text, parses demographics/meds/etc.
Step 2: detect_conflicts   → finds cross-document contradictions
Step 3: escalate           → one flag per conflict cluster
Step 4: reconcile_meds     → compares admission vs discharge medications
Step 5: check_drug_interactions → checks discharge meds against interaction DB
Step 6: escalate           → one flag per MAJOR/MODERATE interaction found
Step 7: check_pending      → identifies pending labs/results
Step 8: escalate (×N)      → one flag per undocumented med change
Step N: build_summary      → assembles final draft
Step N+1: DONE
```

The LLM planner can deviate from this — if ingestion fails it will retry,
if no conflicts are found it skips escalation, etc.

---

## No-Fabrication Guardrail

This is enforced in three places and cannot be disabled by the LLM:

1. **`summary_builder.py`**: Every required field that cannot be found in
   source documents is replaced with the literal string
   `[MISSING — REQUIRES CLINICIAN INPUT]`. Pending results get
   `[PENDING — DO NOT ASSUME VALUE]`. This is hard-coded, not prompted.

2. **`pending_checker.py`**: Any result marked pending is recorded verbatim
   and passed to the summary as-is. The tool never fills in a plausible value.

3. **`_meta.warning`**: Every output contains a machine-readable warning
   confirming the document is a draft for review, `auto_finalized: false`.

The LLM planner only decides *which tools to call and when*. It never
directly writes clinical content into the output.

---

## Conflict & Failure Handling

**Conflicts**: `detect_conflicts` finds disagreements between documents
(different diagnoses, different dates, allergy contradictions). The agent
escalates them as flags. `summary_builder` marks conflicting fields with
`[CONFLICTING DATA — REQUIRES CLINICIAN RESOLUTION]` — it never picks a winner.

**Tool failures**: Wrapped in `try/except` with `RETRY_LIMIT = 3` and
exponential backoff. Failed actions are recorded in `state.failed_actions`
so the planner knows to re-plan around them. The agent never behaves as if
a failed call succeeded.

**Planner failures**: If the LLM returns malformed JSON or the call fails,
the agent adds an ABORT trace step and terminates cleanly.

**Step cap**: Hard cap at 25 steps. If hit, a `SYSTEM_WARNING` flag is added
so clinicians know the summary may be incomplete.

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

This produces `(draft, corrected)` pairs without real clinicians.

### Learning mechanism: Correction Memory

A structured key-value store of `(field, original_draft, clinician_correction)` triples.
Retrieved by string similarity to the current draft. Injected as few-shot examples
into the planner's system prompt for relevant fields.

### Results (3 patients × 3 epochs)

| Metric | Round 1 | Round 6 | Change |
|--------|---------|---------|--------|
| Aggregate reward | 0.63 | 0.78 | +0.15 |
| Edit burden | 0.37 | 0.22 | -0.15 |
| Holdout reward | — | 0.79 | — |

Trend: **improving** across rounds.

### Limitations

**Cold start**: Empty memory gives no uplift. Mitigation: seed with
manually authored gold-standard examples before first deployment.

**Gaming the reward**: An agent that always outputs MISSING everywhere
reduces edit distance by being maximally vague. Mitigation: missing
fields are penalized in reward computation; a fully-MISSING summary scores ~0.

**Style vs accuracy**: The reviewer edits style as well as medicine.
A model learning style improvements earns reward even if clinical content
is wrong. Mitigation in production: separate style-edit and clinical-edit
pairs; weight clinical edits 3×.

**Safety preservation**: Learning only affects prompt injection for style
fields (follow-up wording, condition expansion). The no-fabrication guardrail
is hard-coded in `summary_builder.py`, not in any learnable prompt component.

**Simulated vs real reviewers**: Real clinicians vary, disagree, and make
their own errors. Production use requires IRB approval, annotation guidelines,
and inter-rater reliability checks.

---

## What I'd Do With More Time

1. **Embedding-based retrieval** for correction memory (vs string similarity)
2. **Real PDF generation** via reportlab for more realistic OCR testing
3. **Section-level confidence scores** propagated through to the UI
4. **Structured conflict resolution UI** so clinicians can resolve flags inline
5. **DPO fine-tuning** on (draft, corrected) pairs once N > 100
6. **ICD-10 code validation** for diagnoses
7. **Allergy cross-check** against prescribed medications
