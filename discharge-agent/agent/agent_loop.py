"""
Discharge Summary Agent - Core Loop
====================================
A real agentic loop: the agent reads plans, calls tools, re-plans based on
results, and never fabricates clinical facts.

Design:
  - Planner LLM call  → decide next action
  - Tool executor     → run the chosen tool
  - Evaluator         → did we get what we needed? re-plan if not
  - Hard step cap     → safety ceiling on iterations
  - Trace emitter     → structured log of every decision
"""

import json
import time
import uuid
import copy
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

from tools.pdf_ingestion import PDFIngestionTool
from tools.conflict_detector import ConflictDetectorTool
from tools.med_reconciler import MedReconciliationTool
from tools.drug_interaction import DrugInteractionTool
from tools.escalation import EscalationTool
from tools.summary_builder import SummaryBuilderTool
from tools.pending_checker import PendingCheckerTool

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_STEPS = 25          # Hard iteration cap — agent cannot run forever
RETRY_LIMIT = 3         # Per-tool retry ceiling


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    step: int
    reasoning: str
    action: str
    inputs: dict
    result: Any
    next_decision: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentState:
    patient_id: str
    pdf_paths: list[str]
    extracted_sections: dict = field(default_factory=dict)   # raw text per section type
    structured_data: dict = field(default_factory=dict)      # parsed clinical facts
    conflicts: list[dict] = field(default_factory=list)      # detected contradictions
    pending_items: list[dict] = field(default_factory=list)  # pending labs / results
    flags: list[dict] = field(default_factory=list)          # clinician escalation flags
    med_reconciliation: dict = field(default_factory=dict)   # admission vs discharge meds
    drug_interactions: list[dict] = field(default_factory=list)
    draft_summary: dict = field(default_factory=dict)        # structured output
    completed_actions: list[str] = field(default_factory=list)
    failed_actions: list[str] = field(default_factory=list)
    step: int = 0
    done: bool = False
    abort_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────────────────────────────────────

TOOLS = {
    "ingest_pdfs":          PDFIngestionTool(),
    "detect_conflicts":     ConflictDetectorTool(),
    "reconcile_meds":       MedReconciliationTool(),
    "check_drug_interactions": DrugInteractionTool(),
    "check_pending":        PendingCheckerTool(),
    "escalate":             EscalationTool(),
    "build_summary":        SummaryBuilderTool(),
}

TOOL_DESCRIPTIONS = {
    "ingest_pdfs": "Extract and parse all source PDF documents for this patient. Returns structured sections: demographics, admission note, progress notes, labs, medications, allergies, follow-up. MUST be called first.",
    "detect_conflicts": "Scan extracted data for contradictions between notes (e.g. different diagnoses in different documents). Returns list of conflicts to flag.",
    "reconcile_meds": "Compare admission medications against discharge medications. Identify additions, discontinuations, dose changes, and any changes without documented reason.",
    "check_drug_interactions": "Run discharge medication list through interaction checker. Returns any significant interactions that must be surfaced for clinician review.",
    "check_pending": "Identify any laboratory results, imaging, or consults marked as pending or not yet resulted. These must appear in the output, never be filled in.",
    "escalate": "Create a formal clinician-review flag for a specific concern. Use for: missing required fields, unresolved conflicts, significant drug interactions, undocumented med changes.",
    "build_summary": "Assemble the final structured discharge summary draft from all gathered state. Only call when extraction, conflicts, reconciliation, and pending checks are complete.",
}


# ──────────────────────────────────────────────────────────────────────────────
# LLM planner (calls Anthropic API)
# ──────────────────────────────────────────────────────────────────────────────

def call_planner(state: AgentState, llm_client, trace: list[TraceStep]) -> dict:
    """
    Ask the LLM what to do next given current state.
    Returns: { "reasoning": str, "action": str, "inputs": dict, "done": bool }
    """
    system = """You are the planning component of a medical discharge summary agent.
Your job: decide the NEXT SINGLE action to take based on current state.

CRITICAL RULES:
1. Never fabricate clinical facts. If data is missing, mark it missing.
2. Always call ingest_pdfs first before any other tool.
3. Always call detect_conflicts, reconcile_meds, check_pending before build_summary.
4. Call escalate for: missing required fields, unresolved conflicts, drug interactions, undocumented med changes.
5. Call build_summary only when all checks are complete.
6. If you have nothing left to do, set done=true.

Available tools and when to use them:
""" + "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()) + """

Respond ONLY with valid JSON (no markdown):
{
  "reasoning": "explain your thinking step by step",
  "action": "tool_name or DONE",
  "inputs": { ... tool-specific params ... },
  "done": false
}

If action is DONE, set done=true and action="DONE".
"""

    state_summary = {
        "patient_id": state.patient_id,
        "pdf_paths": state.pdf_paths,
        "completed_actions": state.completed_actions,
        "failed_actions": state.failed_actions,
        "has_extracted_sections": bool(state.extracted_sections),
        "sections_found": list(state.extracted_sections.keys()),
        "structured_data_keys": list(state.structured_data.keys()),
        "num_conflicts": len(state.conflicts),
        "num_pending": len(state.pending_items),
        "num_flags": len(state.flags),
        "med_reconciliation_done": bool(state.med_reconciliation),
        "drug_interactions_checked": bool(state.drug_interactions is not None),
        "draft_summary_built": bool(state.draft_summary),
        "step": state.step,
        "conflicts_summary": state.conflicts[:3],   # avoid huge context
        "pending_summary": state.pending_items[:3],
        "flags_summary": state.flags[:3],
        "med_changes": state.med_reconciliation.get("changes", [])[:5] if state.med_reconciliation else [],
    }

    recent_trace = [
        {"step": t.step, "action": t.action, "result_summary": str(t.result)[:200], "error": t.error}
        for t in trace[-5:]  # last 5 steps only
    ]

    user_msg = f"""Current agent state:
{json.dumps(state_summary, indent=2)}

Recent trace (last 5 steps):
{json.dumps(recent_trace, indent=2)}

What is the next action?"""

    response = llm_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user_msg}]
    )

    raw = response.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Tool executor with retry logic
# ──────────────────────────────────────────────────────────────────────────────

def execute_tool(action: str, inputs: dict, state: AgentState) -> tuple[Any, Optional[str]]:
    """
    Run the chosen tool with retry logic.
    Returns (result, error_message_or_None)
    """
    tool = TOOLS.get(action)
    if not tool:
        return None, f"Unknown tool: {action}"

    last_error = None
    for attempt in range(RETRY_LIMIT):
        try:
            result = tool.run(inputs, state)
            return result, None
        except Exception as e:
            last_error = str(e)
            if attempt < RETRY_LIMIT - 1:
                time.sleep(0.5 * (attempt + 1))   # back-off

    return None, f"Tool {action} failed after {RETRY_LIMIT} attempts: {last_error}"


# ──────────────────────────────────────────────────────────────────────────────
# State updater — apply tool result to state
# ──────────────────────────────────────────────────────────────────────────────

def apply_result(action: str, result: Any, state: AgentState):
    """Merge tool result into agent state."""
    if action == "ingest_pdfs" and result:
        state.extracted_sections = result.get("sections", {})
        state.structured_data = result.get("structured", {})

    elif action == "detect_conflicts" and result:
        state.conflicts = result.get("conflicts", [])

    elif action == "reconcile_meds" and result:
        state.med_reconciliation = result

    elif action == "check_drug_interactions" and result:
        state.drug_interactions = result.get("interactions", [])

    elif action == "check_pending" and result:
        state.pending_items = result.get("pending", [])

    elif action == "escalate" and result:
        state.flags.append(result)

    elif action == "build_summary" and result:
        state.draft_summary = result


# ──────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(patient_id: str, pdf_paths: list[str], llm_client, progress_callback=None) -> dict:
    """
    Run the full agentic loop for a patient.
    Returns: { "summary": dict, "trace": list, "flags": list, "state": dict }
    """
    state = AgentState(patient_id=patient_id, pdf_paths=pdf_paths)
    trace: list[TraceStep] = []

    print(f"\n{'='*60}")
    print(f"AGENT START — Patient: {patient_id}")
    print(f"PDFs: {pdf_paths}")
    print(f"{'='*60}\n")

    while state.step < MAX_STEPS and not state.done:
        state.step += 1
        step_start = time.time()

        if progress_callback:
            progress_callback(state.step, "planning", None)

        # ── 1. Plan ──────────────────────────────────────────────────────────
        try:
            plan = call_planner(state, llm_client, trace)
        except Exception as e:
            trace.append(TraceStep(
                step=state.step,
                reasoning="Planner LLM call failed",
                action="ABORT",
                inputs={},
                result=None,
                next_decision="Aborting due to planner failure",
                error=str(e)
            ))
            state.abort_reason = f"Planner failed: {e}"
            state.done = True
            break

        action = plan.get("action", "DONE")
        inputs = plan.get("inputs", {})
        reasoning = plan.get("reasoning", "")

        print(f"\n[Step {state.step}] Reasoning: {reasoning[:200]}")
        print(f"[Step {state.step}] Action: {action}")

        if progress_callback:
            progress_callback(state.step, action, reasoning)

        # ── 2. Check done ─────────────────────────────────────────────────────
        if action == "DONE" or plan.get("done"):
            trace.append(TraceStep(
                step=state.step,
                reasoning=reasoning,
                action="DONE",
                inputs={},
                result="Agent decided work is complete",
                next_decision="Finalizing output"
            ))
            state.done = True
            break

        # ── 3. Execute tool ───────────────────────────────────────────────────
        result, error = execute_tool(action, inputs, state)
        duration_ms = (time.time() - step_start) * 1000

        if error:
            state.failed_actions.append(f"step{state.step}:{action}")
            next_decision = f"Tool failed — will report failure and re-plan without this result"
            print(f"[Step {state.step}] ERROR: {error}")
        else:
            apply_result(action, result, state)
            state.completed_actions.append(f"step{state.step}:{action}")
            next_decision = f"Tool succeeded — updating state and re-planning"
            print(f"[Step {state.step}] Result: {str(result)[:200]}")

        # ── 4. Record trace ───────────────────────────────────────────────────
        trace.append(TraceStep(
            step=state.step,
            reasoning=reasoning,
            action=action,
            inputs=inputs,
            result=result if not error else None,
            next_decision=next_decision,
            duration_ms=duration_ms,
            error=error
        ))

    # ── Step cap hit ──────────────────────────────────────────────────────────
    if state.step >= MAX_STEPS and not state.done:
        state.abort_reason = f"Hard step cap ({MAX_STEPS}) reached"
        state.flags.append({
            "type": "SYSTEM_WARNING",
            "severity": "HIGH",
            "message": f"Agent hit maximum step limit ({MAX_STEPS}). Summary may be incomplete.",
            "requires_review": True
        })
        print(f"\n⚠️  STEP CAP REACHED at step {state.step}")

    print(f"\n{'='*60}")
    print(f"AGENT DONE — {state.step} steps, {len(state.flags)} flags")
    print(f"{'='*60}\n")

    return {
        "patient_id": patient_id,
        "summary": state.draft_summary,
        "trace": [t.to_dict() for t in trace],
        "flags": state.flags,
        "conflicts": state.conflicts,
        "pending_items": state.pending_items,
        "med_reconciliation": state.med_reconciliation,
        "drug_interactions": state.drug_interactions,
        "step_count": state.step,
        "abort_reason": state.abort_reason,
        "completed_actions": state.completed_actions,
        "failed_actions": state.failed_actions,
    }
