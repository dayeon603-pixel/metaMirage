"""
CognitiveMirage — Cross-Judge Validation (Anthropic-only subset).

Generates raw model responses once for the two Claude models on all 50 tasks,
then judges each response with BOTH claude-sonnet-4-5 and claude-opus-4-5.
Reports weighted Cohen's kappa per rubric dimension and the Pearson
correlation of total_score across the two judges.

This addresses the single-judge limitation on the scorable Anthropic subset;
cross-vendor validation (GPT-4o as second judge) is deferred pending
OpenAI billing.

Output:
  data/cross_judge_records.json  — per-(task, model, judge) records
  data/cross_judge_report.json   — aggregated statistics
"""
from __future__ import annotations
import os, sys, time, json, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from anthropic import Anthropic

from v3_judge_evaluator import (
    EVAL_SYSTEM, build_judge_prompt, call_anthropic,
    JUDGE_RUBRIC_SYSTEM, JUDGE_ABSTAIN_SYSTEM, JUDGE_EXPERTISE_SYSTEM,
)

MODELS = ["claude-opus-4-5", "claude-sonnet-4-5"]
JUDGES = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus":   "claude-opus-4-20250514",
}
TASKS_FILE = Path("v3_tasks_50.json")
OUT_RECORDS = Path("data/cross_judge_records.json")
OUT_REPORT  = Path("data/cross_judge_report.json")


def _judge_system(mode):
    return {"rubric": JUDGE_RUBRIC_SYSTEM,
            "abstain_binary": JUDGE_ABSTAIN_SYSTEM,
            "expertise_inverted": JUDGE_EXPERTISE_SYSTEM}.get(mode, JUDGE_RUBRIC_SYSTEM)


def _parse_judge(jr_text, mode):
    parsed = json.loads(jr_text.strip())
    out = {"reasoning": parsed.get("reasoning", "")}
    if mode == "rubric":
        out.update(
            trap_detection=int(parsed["trap_detection"]),
            conf_appropriate=int(parsed["confidence_appropriateness"]),
            answer_quality=int(parsed["answer_quality"]),
        )
        out["total"] = (out["trap_detection"] + out["conf_appropriate"] + out["answer_quality"]) / 9.0
    elif mode == "abstain_binary":
        out.update(
            abstain_score=int(parsed["abstain_score"]),
            answer_quality=int(parsed["answer_quality"]),
        )
        out["total"] = (out["abstain_score"] * 0.6 + out["answer_quality"] * 0.4) / 3.0
    elif mode == "expertise_inverted":
        out.update(
            metacognitive_flag=int(parsed["metacognitive_flag"]),
            confidence_calibration=int(parsed["confidence_calibration"]),
        )
        out["total"] = (out["metacognitive_flag"] * 0.7 + out["confidence_calibration"] * 0.3) / 3.0
    return out


def _retry(fn, retries=2, label=""):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                print(f"  [FAIL {label}] {e}")
                return None
            time.sleep(2 ** attempt)


def run_one(task, model, ac):
    """Call model once; judge the response with both judges."""
    mode = task.get("scoring_mode", "rubric")
    resp = _retry(lambda: call_anthropic(model, task["prompt"], EVAL_SYSTEM, ac),
                  label=f"model={model} task={task['task_id']}")
    if resp is None:
        return None

    jp = build_judge_prompt(task, resp)
    sys_p = _judge_system(mode)
    judged = {}
    for jkey, jmodel in JUDGES.items():
        jr = _retry(lambda jm=jmodel: call_anthropic(jm, jp, sys_p, ac),
                    label=f"judge={jkey} task={task['task_id']}")
        if jr is None:
            judged[jkey] = None
            continue
        try:
            judged[jkey] = _parse_judge(jr, mode)
        except Exception as e:
            print(f"  [PARSE FAIL {jkey}] {task['task_id']}: {e}")
            judged[jkey] = None
    return {
        "task_id":      task["task_id"],
        "family":       task["family"],
        "variant":      task["variant"],
        "scoring_mode": mode,
        "model":        model,
        "response":     resp,
        "judged":       judged,
    }


# ── STATISTICS ──

def weighted_kappa(a, b, ratings=(0, 1, 2, 3)):
    """Cohen's weighted (quadratic) kappa for ordinal ratings."""
    n = len(a)
    if n == 0: return None
    k = len(ratings)
    r2i = {v: i for i, v in enumerate(ratings)}
    O = [[0] * k for _ in range(k)]
    for ai, bi in zip(a, b):
        if ai is None or bi is None: continue
        O[r2i[ai]][r2i[bi]] += 1
    na = sum(sum(r) for r in O)
    if na == 0: return None
    row_sum = [sum(r) for r in O]
    col_sum = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            e_ij = row_sum[i] * col_sum[j] / na
            num += w * O[i][j]
            den += w * e_ij
    if den == 0: return 1.0
    return round(1 - num / den, 4)


def pearson_total(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3: return None
    xs, ys = zip(*pairs)
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(num / den, 4) if den else None


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")
    OUT_RECORDS.parent.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(TASKS_FILE.read_text())
    ac = Anthropic()

    total = len(tasks) * len(MODELS)
    print(f"Cross-judge: {len(MODELS)} models × {len(tasks)} tasks = {total} calls × (1 + {len(JUDGES)}) inference passes")

    records = []
    done = [0]
    for model in MODELS:
        print(f"\n── {model} ──")
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(run_one, t, model, ac): t["task_id"] for t in tasks}
            for f in as_completed(futs):
                rec = f.result()
                done[0] += 1
                if rec:
                    records.append(rec)
                    s1 = rec["judged"].get("sonnet", {}) or {}
                    s2 = rec["judged"].get("opus", {}) or {}
                    print(f"  [{done[0]:>3}/{total}] {rec['task_id']} {rec['variant']:7s} "
                          f"sonnet={s1.get('total', float('nan')):.3f} opus={s2.get('total', float('nan')):.3f}")
                else:
                    print(f"  [{done[0]:>3}/{total}] FAIL")

    OUT_RECORDS.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} records → {OUT_RECORDS}")

    # ── Aggregate statistics ──
    report = {"n_records": len(records), "kappa": {}, "pearson_total": {}, "per_mode": {}}

    # Group dimensions by mode
    dims = {
        "rubric":             ["trap_detection", "conf_appropriate", "answer_quality"],
        "abstain_binary":     ["abstain_score", "answer_quality"],
        "expertise_inverted": ["metacognitive_flag", "confidence_calibration"],
    }

    for mode, fields in dims.items():
        mode_recs = [r for r in records if r["scoring_mode"] == mode
                     and r["judged"].get("sonnet") and r["judged"].get("opus")]
        if not mode_recs: continue
        report["per_mode"][mode] = {"n": len(mode_recs), "kappa": {}, "pearson_total": None}
        for field in fields:
            a = [r["judged"]["sonnet"].get(field) for r in mode_recs]
            b = [r["judged"]["opus"].get(field) for r in mode_recs]
            report["per_mode"][mode]["kappa"][field] = weighted_kappa(a, b)
        ta = [r["judged"]["sonnet"]["total"] for r in mode_recs]
        tb = [r["judged"]["opus"]["total"] for r in mode_recs]
        report["per_mode"][mode]["pearson_total"] = pearson_total(ta, tb)

    # Overall total-score agreement
    all_valid = [r for r in records if r["judged"].get("sonnet") and r["judged"].get("opus")]
    ta = [r["judged"]["sonnet"]["total"] for r in all_valid]
    tb = [r["judged"]["opus"]["total"] for r in all_valid]
    report["overall_pearson_total"] = pearson_total(ta, tb)
    report["overall_mean_abs_diff"] = round(sum(abs(a - b) for a, b in zip(ta, tb)) / len(ta), 4) if ta else None
    report["n_judged_both"] = len(all_valid)

    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"\n=== Cross-Judge Report ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved → {OUT_REPORT}")


if __name__ == "__main__":
    main()
