"""
Main Runner
===========
Entry point for the discharge summary agent system.

Usage:
  python main.py --patient PT-001          # Run agent on one patient
  python main.py --all                     # Run all patients
  python main.py --all --part2             # Include Part 2 learning loop
  python main.py --demo                    # Demo mode with synthetic data

Outputs:
  output/<patient_id>/discharge_summary.json   — structured summary
  output/<patient_id>/trace.json               — step trace
  output/<patient_id>/flags.json               — clinician flags
  output/part2_results.json                    — learning curve (if --part2)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_patients import generate_all_patients, PATIENTS


def get_llm_client():
    """Initialize LLM client. Reads API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Set it with: export ANTHROPIC_API_KEY=your_key_here"
        )
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )


def run_single_patient(patient_id: str, pdf_paths: list, client, output_dir: str) -> dict:
    """Run agent on a single patient and save outputs."""
    from agent.agent_loop import run_agent

    print(f"\n{'='*60}")
    print(f"Processing Patient: {patient_id}")
    print(f"Documents: {len(pdf_paths)} files")
    print(f"{'='*60}")

    start = time.time()
    result = run_agent(
        patient_id=patient_id,
        pdf_paths=pdf_paths,
        llm_client=client,
    )
    elapsed = time.time() - start

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Summary
    summary_path = os.path.join(output_dir, "discharge_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result["summary"], f, indent=2)

    # Trace
    trace_path = os.path.join(output_dir, "trace.json")
    with open(trace_path, "w") as f:
        json.dump(result["trace"], f, indent=2)

    # Flags
    flags_path = os.path.join(output_dir, "flags.json")
    with open(flags_path, "w") as f:
        json.dump({
            "flags": result["flags"],
            "conflicts": result["conflicts"],
            "pending_items": result["pending_items"],
            "drug_interactions": result["drug_interactions"],
        }, f, indent=2)

    # Human-readable trace
    trace_txt_path = os.path.join(output_dir, "trace_readable.txt")
    with open(trace_txt_path, "w") as f:
        f.write(f"AGENT TRACE — Patient {patient_id}\n")
        f.write(f"{'='*60}\n\n")
        for step in result["trace"]:
            f.write(f"[Step {step['step']}] Action: {step['action']}\n")
            f.write(f"  Reasoning: {step['reasoning'][:300]}\n")
            if step.get("error"):
                f.write(f"  ERROR: {step['error']}\n")
            else:
                result_str = str(step.get("result", ""))[:200]
                f.write(f"  Result: {result_str}\n")
            f.write(f"  Next: {step['next_decision']}\n")
            f.write(f"  Duration: {step.get('duration_ms', 0):.0f}ms\n\n")

    print(f"\n✓ Completed in {elapsed:.1f}s")
    print(f"  Steps: {result['step_count']}")
    print(f"  Flags: {len(result['flags'])}")
    print(f"  Conflicts: {len(result['conflicts'])}")
    print(f"  Pending items: {len(result['pending_items'])}")
    print(f"  Drug interactions: {len(result.get('drug_interactions') or [])}")
    if result.get("abort_reason"):
        print(f"  ⚠ Abort reason: {result['abort_reason']}")
    print(f"\n  Outputs saved to: {output_dir}")

    return result


def run_part2_learning(all_results: list, output_dir: str):
    """Run Part 2: learning loop on collected drafts."""
    from agent.learning_loop import (
        SimulatedReviewer, CorrectionMemory, LearningLoop, LIMITATIONS
    )

    print(f"\n{'='*60}")
    print("PART 2: Learning from Doctor Edits")
    print(f"{'='*60}")

    reviewer = SimulatedReviewer()
    memory = CorrectionMemory(max_entries=50)
    loop = LearningLoop(memory=memory, reviewer=reviewer)

    # Build held-out set (last 1/3 of results)
    n = len(all_results)
    train_results = all_results[:max(1, n * 2 // 3)]
    holdout_results = all_results[max(1, n * 2 // 3):]

    print(f"\nTraining set: {len(train_results)} patients")
    print(f"Held-out set: {len(holdout_results)} patients")

    # For demonstration with limited data: repeat patients across rounds
    ROUNDS = 9
    round_num = 0
    all_round_results = []

    # Simulate multiple rounds by re-running same patients (realistic with real data would be different patients)
    for epoch in range(3):
        for result in train_results:
            if not result.get("summary"):
                continue
            round_num += 1
            patient_meta = {
                "resolution_hint": None  # Reviewer uses its own policy
            }

            round_result = loop.run_round(
                round_num=round_num,
                patient_id=result["patient_id"],
                draft=result["summary"],
                patient_metadata=patient_meta,
            )
            all_round_results.append(round_result)

            reward = round_result["reward"]["aggregate_reward"]
            num_edits = len(round_result["edits"])
            print(f"  Round {round_num} | Patient {result['patient_id']} | "
                  f"Reward: {reward:.3f} | Edits: {num_edits} | "
                  f"Memory: {round_result['memory_size']}")

    # Held-out evaluation
    holdout_rewards = []
    print(f"\n--- Held-out Evaluation ---")
    for result in holdout_results:
        if not result.get("summary"):
            continue
        round_num += 1
        round_result = loop.run_round(
            round_num=round_num,
            patient_id=result["patient_id"] + "_holdout",
            draft=result["summary"],
            patient_metadata={},
        )
        reward = round_result["reward"]["aggregate_reward"]
        holdout_rewards.append(reward)
        print(f"  Holdout | Patient {result['patient_id']} | Reward: {reward:.3f}")

    curve = loop.get_improvement_curve()

    print(f"\n{'='*40}")
    print(f"LEARNING RESULTS:")
    print(f"  First-round reward: {curve.get('first_round_reward', 'N/A')}")
    print(f"  Latest reward:      {curve.get('latest_round_reward', 'N/A')}")
    print(f"  Reward improvement: {curve.get('improvement_in_reward', 'N/A')}")
    print(f"  Edit burden reduction: {curve.get('edit_burden_reduction', 'N/A')}")
    print(f"  Trend: {curve.get('trend', 'N/A')}")
    if holdout_rewards:
        print(f"  Holdout avg reward: {sum(holdout_rewards)/len(holdout_rewards):.3f}")
    print(f"{'='*40}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    loop.save_results(os.path.join(output_dir, "part2_learning_results.json"))
    memory.save(os.path.join(output_dir, "correction_memory.json"))

    # Save limitations discussion
    with open(os.path.join(output_dir, "limitations.json"), "w") as f:
        json.dump(LIMITATIONS, f, indent=2)

    print(f"\n  Part 2 results saved to: {output_dir}")
    return curve


def demo_mode(output_dir: str):
    """
    Demo mode: runs the agent using the mock LLM planner (no API key needed).
    Uses a rule-based planner to simulate the agent loop for demonstration.
    """
    from data.generate_patients import generate_all_patients
    import sys, os

    print("\n" + "="*60)
    print("DEMO MODE — Running agent with rule-based planner")
    print("(No API key required)")
    print("="*60)

    # Generate patient data
    base_dir = os.path.join(os.path.dirname(__file__), "data", "patients")
    registry = generate_all_patients(base_dir)

    # Use mock client
    class MockClient:
        """Rule-based planner for demo mode."""
        def __init__(self):
            self._step = 0
            self._plan = [
                ("ingest_pdfs", {}),
                ("detect_conflicts", {}),
                ("reconcile_meds", {}),
                ("check_drug_interactions", {}),
                ("check_pending", {}),
                ("escalate", {
                    "type": "MISSING_REQUIRED_FIELD",
                    "severity": "HIGH",
                    "message": "Discharge condition not documented in source notes",
                    "field": "discharge_condition"
                }),
                ("build_summary", {}),
            ]

        class messages:
            @staticmethod
            def create(**kwargs):
                return None  # Not used in mock mode

    # Monkey-patch planner for demo
    import agent.agent_loop as al

    def mock_planner(state, llm_client, trace):
        # Build clean set of which tool types have been completed
        # Use step:action format from completed_actions e.g. "step1:ingest_pdfs"
        done = set(a.split(":")[1] for a in state.completed_actions)
        num_escalations = sum(1 for a in state.completed_actions if a.split(":")[1] == "escalate")

        # Step 1: Always ingest first
        if "ingest_pdfs" not in done:
            return {"reasoning": "First step: ingest all patient PDFs", "action": "ingest_pdfs", "inputs": {"pdf_paths": state.pdf_paths}, "done": False}

        # Step 2: Detect conflicts
        if "detect_conflicts" not in done:
            return {"reasoning": "Check for cross-document contradictions", "action": "detect_conflicts", "inputs": {}, "done": False}

        # Step 3: Escalate conflicts (once only)
        if state.conflicts and num_escalations == 0:
            return {
                "reasoning": f"Found {len(state.conflicts)} conflict(s) — escalating all as single flag",
                "action": "escalate",
                "inputs": {
                    "type": "DIAGNOSIS_CONFLICT", "severity": "HIGH",
                    "message": f"{len(state.conflicts)} data conflict(s): " + "; ".join(c.get("description","") for c in state.conflicts[:2]),
                    "field": "principal_diagnosis", "details": {"conflicts": state.conflicts[:3]}
                },
                "done": False
            }

        # Step 4: Reconcile meds
        if "reconcile_meds" not in done:
            return {"reasoning": "Compare admission vs discharge medications", "action": "reconcile_meds", "inputs": {}, "done": False}

        # Step 5: Check drug interactions
        if "check_drug_interactions" not in done:
            return {"reasoning": "Check discharge medications for interactions", "action": "check_drug_interactions", "inputs": {}, "done": False}

        # Step 6: Escalate drug interactions (once only, only if MAJOR found)
        ix_escalated = any(f.get("type") == "DRUG_INTERACTION" for f in state.flags)
        if not ix_escalated and state.drug_interactions:
            major = [i for i in state.drug_interactions if i.get("severity") == "MAJOR"]
            if major:
                return {
                    "reasoning": f"Found {len(major)} MAJOR drug interaction(s) — escalating as CRITICAL flag",
                    "action": "escalate",
                    "inputs": {
                        "type": "DRUG_INTERACTION", "severity": "CRITICAL",
                        "message": "; ".join(i["flag_message"] for i in major[:3]),
                        "field": "medications", "details": {"interactions": major}
                    },
                    "done": False
                }

        # Step 7: Check pending results
        if "check_pending" not in done:
            return {"reasoning": "Find all pending/outstanding results", "action": "check_pending", "inputs": {}, "done": False}

        # Step 8: Escalate undocumented med changes (one per flagged change, capped)
        if state.med_reconciliation:
            changes_to_flag = state.med_reconciliation.get("changes_requiring_review", [])
            med_flags_so_far = sum(1 for f in state.flags if f.get("type") == "UNDOCUMENTED_MED_CHANGE")
            if med_flags_so_far < len(changes_to_flag):
                change = changes_to_flag[med_flags_so_far]
                return {
                    "reasoning": f"Undocumented med change #{med_flags_so_far+1}: {change.get('drug')} — escalating",
                    "action": "escalate",
                    "inputs": {
                        "type": "UNDOCUMENTED_MED_CHANGE", "severity": "HIGH",
                        "message": change.get("flag_reason", "Undocumented medication change"),
                        "field": "medications", "details": change
                    },
                    "done": False
                }

        # Step 9: Build summary once all checks done
        if "build_summary" not in done:
            return {"reasoning": "All checks complete — assembling final discharge summary draft", "action": "build_summary", "inputs": {}, "done": False}

        # Done
        return {"reasoning": "Summary built — all flags raised — done", "action": "DONE", "inputs": {}, "done": True}

    al.call_planner = mock_planner

    all_results = []
    for pid, info in registry.items():
        result = run_single_patient(
            patient_id=pid,
            pdf_paths=info["files"],
            client=None,
            output_dir=os.path.join(output_dir, pid)
        )
        result["patient_id"] = pid
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Discharge Summary Agent")
    parser.add_argument("--patient", help="Patient ID to process (e.g. PT-001)")
    parser.add_argument("--all", action="store_true", help="Process all patients")
    parser.add_argument("--part2", action="store_true", help="Run Part 2 learning loop")
    parser.add_argument("--demo", action="store_true", help="Demo mode (no API key needed)")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        all_results = demo_mode(args.output)
        if args.part2:
            run_part2_learning(all_results, args.output)
        return

    # Generate patient data if not present
    base_dir = os.path.join(os.path.dirname(__file__), "data", "patients")
    registry_path = os.path.join(base_dir, "registry.json")

    if not os.path.exists(registry_path):
        print("Generating synthetic patient data...")
        from data.generate_patients import generate_all_patients
        registry = generate_all_patients(base_dir)
    else:
        with open(registry_path) as f:
            registry = json.load(f)

    client = get_llm_client()
    all_results = []

    if args.patient:
        if args.patient not in registry:
            print(f"Patient {args.patient} not found. Available: {list(registry.keys())}")
            sys.exit(1)
        info = registry[args.patient]
        result = run_single_patient(
            patient_id=args.patient,
            pdf_paths=info["files"],
            client=client,
            output_dir=os.path.join(args.output, args.patient)
        )
        result["patient_id"] = args.patient
        all_results.append(result)

    elif args.all:
        for pid, info in registry.items():
            result = run_single_patient(
                patient_id=pid,
                pdf_paths=info["files"],
                client=client,
                output_dir=os.path.join(args.output, pid)
            )
            result["patient_id"] = pid
            all_results.append(result)
    else:
        parser.print_help()
        return

    if args.part2 and all_results:
        run_part2_learning(all_results, args.output)


if __name__ == "__main__":
    main()
