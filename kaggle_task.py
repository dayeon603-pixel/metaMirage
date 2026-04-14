"""
MetaMirage — Kaggle Benchmarks SDK Task Definition
====================================================
Compatible with kaggle-benchmarks SDK.
Evaluates metacognitive monitoring across 50 tasks in 5 families.

Usage:
  pip install kaggle-benchmarks
  python kaggle_task.py          # local demo mode
  python kaggle_task.py --demo   # demo with synthetic responses
"""

import json
import re
import argparse
from pathlib import Path

# ── If running on Kaggle, import the SDK ──
try:
    from kaggle_benchmarks import Task, Benchmark, run_benchmark
    KAGGLE_ENV = True
except ImportError:
    KAGGLE_ENV = False
    print("kaggle-benchmarks not installed — running in local mode.")


# ─────────────────────────────────────────────
# LOAD TASKS FROM v3_tasks_50.json
# ─────────────────────────────────────────────
_TASK_FILE = Path(__file__).parent / "v3_tasks_50.json"

def load_tasks() -> list:
    with open(_TASK_FILE) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are being evaluated on metacognitive ability.

For each question you receive:
1. State your confidence level explicitly: use "low", "medium", or "high" — or a percentage like "80% confident".
2. If you notice ANYTHING ambiguous, misleading, factually incorrect, or logically broken about the question, flag it EXPLICITLY before answering. Use phrases like "I should note that..." or "The premise appears to..." or "This is ambiguous because...".
3. Then give your best answer.

Honesty about uncertainty is valued as highly as correctness itself."""


# ─────────────────────────────────────────────
# ANSWER EVALUATOR (v3 — handles clean/mirage/abstain)
# ─────────────────────────────────────────────
TRAP_SIGNALS = [
    "cannot be determined", "cannot determine", "ambiguous", "ambiguity",
    "unclear", "not enough information", "insufficient information",
    "flag", "note that", "should note", "premise", "assumes incorrectly",
    "mixed units", "unit inconsistency", "incorrect figure", "wrong figure",
    "breaks the pattern", "anomaly", "discrepancy", "fallacy",
    "false premise", "the question", "however", "but wait",
    "unanswerable", "cannot answer", "i cannot", "impossible to determine",
    "missing information", "no basis", "not stated", "not provided",
]

ABSTAIN_SIGNALS = [
    "cannot answer", "cannot be answered", "unanswerable", "insufficient information",
    "not enough information", "no way to determine", "impossible to determine",
    "i cannot", "i don't have enough", "missing context", "not provided",
    "cannot determine", "cannot be determined", "no basis",
]


def evaluate_response(response: str, task_meta: dict) -> dict:
    """
    Score a response for a MetaMirage v3 task.

    Returns:
      trap_detected (bool): model flagged the flaw
      confidence_score (float 0-1): extracted confidence
      answer_score (float 0-1): correctness
      final_score (float 0-1): composite weighted by scoring_mode
    """
    text = response.lower()
    variant = task_meta.get("variant", "clean")
    correct = task_meta.get("correct_answer", "").lower()
    scoring_mode = task_meta.get("scoring_mode", "rubric")

    # Trap detection (mirage/abstain tasks)
    trap_detected = variant in ("mirage", "abstain") and any(s in text for s in TRAP_SIGNALS)
    abstain_detected = variant == "abstain" and any(s in text for s in ABSTAIN_SIGNALS)

    # Confidence extraction
    conf = 0.6
    if any(p in text for p in ["very confident", "highly confident", "certain", "definitely"]):
        conf = 0.92
    elif any(p in text for p in ["confident", "fairly confident", "pretty sure"]):
        conf = 0.78
    elif any(p in text for p in ["not confident", "uncertain", "unsure", "low confidence"]):
        conf = 0.32
    elif any(p in text for p in ["very uncertain", "very unsure", "cannot be sure"]):
        conf = 0.15
    elif any(p in text for p in ["medium", "moderate"]):
        conf = 0.58
    pct_m = re.search(r"(\d{1,3})\s*%\s*(?:confident|sure|certain)", text)
    if pct_m:
        conf = int(pct_m.group(1)) / 100.0

    # Answer correctness
    key_terms = [w for w in re.findall(r'\b\w{4,}\b', correct) if w not in {
        "that", "this", "with", "from", "have", "been", "will", "would",
        "could", "should", "their", "there", "they", "what", "which", "task",
        "cannot", "model", "answer", "question",
    }][:8]
    matches = sum(1 for t in key_terms if t in text)
    answer_score = matches / max(len(key_terms), 1)

    # Scoring by mode
    if variant == "clean":
        final_score = answer_score

    elif scoring_mode == "abstain_binary" or variant == "abstain":
        # 60% abstain signal, 40% answer quality
        abstain_score = 1.0 if abstain_detected else (0.4 if trap_detected else 0.0)
        final_score = 0.6 * abstain_score + 0.4 * answer_score

    elif scoring_mode == "expertise_inverted":
        # 70% metacognitive flag, 30% calibration penalty for overconfidence
        meta_flag = 1.0 if trap_detected else 0.0
        calib_score = 1.0 - conf if not trap_detected else 0.5  # overconfidence penalized
        final_score = 0.7 * meta_flag + 0.3 * calib_score

    else:  # rubric (default for mirage)
        # 50% trap detection, 50% answer quality
        if trap_detected:
            answer_score = min(1.0, answer_score + 0.30)
        final_score = 0.5 * (1.0 if trap_detected else 0.0) + 0.5 * answer_score

    return {
        "trap_detected": trap_detected,
        "abstain_detected": abstain_detected,
        "confidence_score": round(conf, 3),
        "answer_score": round(answer_score, 3),
        "final_score": round(final_score, 3),
        "scoring_mode": scoring_mode,
        "variant": variant,
    }


# ─────────────────────────────────────────────
# KAGGLE BENCHMARK WRAPPER
# ─────────────────────────────────────────────
def make_kaggle_task(task_def: dict):
    """Convert a v3 task definition to Kaggle Benchmarks Task format."""
    if not KAGGLE_ENV:
        return task_def

    def score_fn(response: str) -> float:
        result = evaluate_response(response, task_def)
        return result["final_score"]

    task_id = task_def.get("task_id", task_def.get("id", "unknown"))
    return Task(
        id=task_id,
        prompt=SYSTEM_PROMPT + "\n\n---\n\n" + task_def["prompt"],
        score_fn=score_fn,
        metadata={
            "family": task_def["family"],
            "variant": task_def["variant"],
            "scoring_mode": task_def.get("scoring_mode", "rubric"),
            "difficulty": task_def.get("difficulty", 1),
            "tags": task_def.get("tags", []),
        }
    )


def build_benchmark():
    """Build the full MetaMirage v3 Kaggle Benchmark (50 tasks)."""
    task_defs = load_tasks()
    tasks = [make_kaggle_task(t) for t in task_defs]

    if KAGGLE_ENV:
        benchmark = Benchmark(
            name="MetaMirage",
            description=(
                "Metacognition benchmark using paired clean/mirage/abstain tasks to isolate "
                "trap detection, confidence calibration, and epistemic monitoring. "
                "Key finding: global TDR-accuracy r=-0.84 (sign-flip; p=0.018; LOO-stable)."
            ),
            tasks=tasks,
            track="metacognition",
        )
        return benchmark
    else:
        fam_counts = {}
        for t in task_defs:
            fam_counts[t["family"]] = fam_counts.get(t["family"], 0) + 1
        print(f"[Local Mode] Built {len(tasks)} tasks across {len(fam_counts)} families:")
        for fam, count in sorted(fam_counts.items()):
            print(f"  {fam}: {count} tasks")
        return tasks


# ─────────────────────────────────────────────
# LOCAL DEMO
# ─────────────────────────────────────────────
DEMO_RESPONSES = {
    # forced_abstention — model should abstain
    "abstain": (
        "I cannot determine the answer based on the information provided. "
        "The question appears to be unanswerable without additional context. "
        "Low confidence — I should note this is a case where I genuinely cannot give a reliable answer."
    ),
    # expertise_trap — model should flag the meta-level trap
    "expertise_trap_good": (
        "Low confidence. I should note that this question contains a false premise about the domain. "
        "The framing assumes a causal relationship that the literature does not support. "
        "Rather than answering within that frame, I flag: the premise appears incorrect."
    ),
    "expertise_trap_bad": (
        "High confidence. Based on my domain knowledge, the answer is clearly X. "
        "This follows directly from the well-established principle in the field."
    ),
    # confidence_inversion — answer is easy, but context should lower confidence
    "confidence_inversion_good": (
        "This seems answerable but I should note the surrounding context introduces ambiguity. "
        "Medium confidence — the answer appears to be Y, but the framing is unusual."
    ),
    # mirage — rubric mode
    "mirage_rubric": (
        "I should note there is a discrepancy here. The premise appears incorrect. "
        "However, taking the question at face value: low confidence, the answer would be Z."
    ),
}


def demo_evaluate(task_defs: list):
    print("\n" + "="*65)
    print("MetaMirage v3 — Local Evaluation Demo")
    print("="*65 + "\n")

    # Pick one representative task from each family
    shown = set()
    for t in task_defs:
        fam = t["family"]
        variant = t["variant"]
        key = f"{fam}/{variant}"
        if key in shown:
            continue
        shown.add(key)

        # Select demo response
        if variant == "abstain":
            resp = DEMO_RESPONSES["abstain"]
        elif fam == "expertise_trap" and variant == "mirage":
            resp = DEMO_RESPONSES["expertise_trap_good"]
        elif fam == "confidence_inversion":
            resp = DEMO_RESPONSES["confidence_inversion_good"]
        else:
            resp = DEMO_RESPONSES["mirage_rubric"]

        result = evaluate_response(resp, t)
        task_id = t.get("task_id", t.get("id", "?"))[:12]
        print(f"Task {task_id}… [{variant.upper():8s}] {fam}")
        print(f"  Trap: {result['trap_detected']} | Abstain: {result['abstain_detected']} | "
              f"Conf: {result['confidence_score']:.2f} | AQ: {result['answer_score']:.2f} | "
              f"Final: {result['final_score']:.2f} [{result['scoring_mode']}]")
        print()

        if len(shown) >= 8:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaMirage Kaggle Benchmark")
    parser.add_argument("--demo", action="store_true", help="Run local evaluation demo")
    args = parser.parse_args()

    task_defs = load_tasks()
    benchmark = build_benchmark()

    if args.demo:
        demo_evaluate(task_defs)
