"""
Re-judge haiku's 50 responses with claude-opus-4-5 as second judge.

If haiku retains its #1 ranking under opus judgment, the self-preference
attack ("haiku benefits from Anthropic-vendor overlap") fails.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from anthropic import Anthropic

from v3_judge_evaluator import (
    EVAL_SYSTEM, build_judge_prompt, call_anthropic,
    JUDGE_RUBRIC_SYSTEM, JUDGE_ABSTAIN_SYSTEM, JUDGE_EXPERTISE_SYSTEM,
)
from v3_statistical_analysis import profile_model
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

HAIKU_RECORDS = Path("data/haiku_records.json")
OPUS_JUDGE    = "claude-opus-4-20250514"
OUT           = Path("data/haiku_rejudged_opus.json")


def judge_system(mode):
    return {"rubric": JUDGE_RUBRIC_SYSTEM,
            "abstain_binary": JUDGE_ABSTAIN_SYSTEM,
            "expertise_inverted": JUDGE_EXPERTISE_SYSTEM}[mode]


def rejudge_one(rec, ac):
    # Reconstruct task dict from the original record
    task = {
        "task_id":      rec["task_id"],
        "family":       rec["family"],
        "variant":      rec["variant"],
        "prompt":       "",
        "correct_answer": "",
        "mirage_signal":  "",
        "scoring_mode": rec["scoring_mode"],
    }
    # We need prompt + correct_answer + mirage_signal from v3_tasks_50.json
    t_by_id = {t["task_id"]: t for t in json.load(open("v3_tasks_50.json"))}
    t_orig = t_by_id[rec["task_id"]]
    task.update(prompt=t_orig["prompt"],
                correct_answer=t_orig.get("correct_answer", ""),
                mirage_signal=t_orig.get("mirage_signal", ""))

    jp = build_judge_prompt(task, rec["raw_response"])
    sys_p = judge_system(rec["scoring_mode"])
    for attempt in range(3):
        try:
            jr = call_anthropic(OPUS_JUDGE, jp, sys_p, ac)
            parsed = json.loads(jr.strip())
            break
        except Exception as e:
            if attempt == 2: return None
            time.sleep(2 ** attempt)

    out = {k: rec[k] for k in ["task_id", "family", "subfamily", "variant",
                                "scoring_mode", "model", "raw_response",
                                "latency_s"]}
    out["judge_error"] = False
    mode = rec["scoring_mode"]
    if mode == "rubric":
        td = int(parsed["trap_detection"])
        ca = int(parsed["confidence_appropriateness"])
        aq = int(parsed["answer_quality"])
        out.update(trap_detection=td, conf_appropriate=ca, answer_quality=aq,
                   total_score=round((td + ca + aq) / 9.0, 4))
    elif mode == "abstain_binary":
        ab = int(parsed["abstain_score"]); aq = int(parsed["answer_quality"])
        out.update(abstain_score=ab, answer_quality=aq,
                   total_score=round((ab * 0.6 + aq * 0.4) / 3.0, 4))
    elif mode == "expertise_inverted":
        mf = int(parsed["metacognitive_flag"]); cc = int(parsed["confidence_calibration"])
        out.update(metacognitive_flag=mf, confidence_calibration=cc,
                   total_score=round((mf * 0.7 + cc * 0.3) / 3.0, 4))
    out["judge_reasoning"] = parsed.get("reasoning", "")
    return out


def main():
    recs = json.loads(HAIKU_RECORDS.read_text())
    print(f"Re-judging {len(recs)} haiku responses with {OPUS_JUDGE}")
    ac = Anthropic()
    out = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(rejudge_one, r, ac): r["task_id"] for r in recs}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if res:
                out.append(res)
                print(f"  [{i:>2}/{len(recs)}] {res['task_id']} total={res['total_score']:.3f}")
            else:
                print(f"  [{i:>2}/{len(recs)}] FAIL")
    Path("data").mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {OUT}")

    # Profile haiku under opus judge
    haiku_profile_opus = profile_model(out, "claude-haiku-4-5-20251001")
    print(f"\n=== Haiku profile under OPUS judge ===")
    print(f"  MI   = {haiku_profile_opus['metacognitive_index']:.4f}")
    print(f"  TDR  = {haiku_profile_opus['tdr_global']:.4f}")
    print(f"  acc  = {haiku_profile_opus['aq_clean']:.4f}")
    print(f"  calΔ = {haiku_profile_opus['calib_delta']:+.4f}")

    # Compare to sonnet judge (existing v3_analysis.json)
    a = json.loads(Path("v3_analysis.json").read_text())
    p_son = a["profiles"]["claude-haiku-4-5-20251001"]
    print(f"\n=== Haiku profile under SONNET judge (current) ===")
    print(f"  MI   = {p_son['metacognitive_index']:.4f}")
    print(f"  TDR  = {p_son['tdr_global']:.4f}")
    print(f"  acc  = {p_son['aq_clean']:.4f}")
    print(f"  calΔ = {p_son['calib_delta']:+.4f}")

    # Would haiku still be #1?
    print(f"\n=== Leaderboard ranking haiku under each judge ===")
    mi_vals = {m: p['metacognitive_index'] for m, p in a['profiles'].items()}
    mi_vals_opus_haiku = dict(mi_vals)
    mi_vals_opus_haiku["claude-haiku-4-5-20251001"] = haiku_profile_opus["metacognitive_index"]
    rank_sonnet = sorted(mi_vals.items(), key=lambda x: -x[1])
    rank_opus = sorted(mi_vals_opus_haiku.items(), key=lambda x: -x[1])
    print("Under Sonnet judge:")
    for i, (m, mi) in enumerate(rank_sonnet, 1): print(f"  {i}. {m:28s} MI={mi:.4f}")
    print("Under Opus judge (for haiku only; others unchanged):")
    for i, (m, mi) in enumerate(rank_opus, 1): print(f"  {i}. {m:28s} MI={mi:.4f}")

    # Write a mini-report
    report = {
        "n_rejudged": len(out),
        "haiku_mi_sonnet_judge": p_son["metacognitive_index"],
        "haiku_mi_opus_judge":   haiku_profile_opus["metacognitive_index"],
        "haiku_tdr_sonnet_judge": p_son["tdr_global"],
        "haiku_tdr_opus_judge":   haiku_profile_opus["tdr_global"],
        "haiku_rank_sonnet": [i for i, (m, _) in enumerate(rank_sonnet, 1)
                              if m == "claude-haiku-4-5-20251001"][0],
        "haiku_rank_opus":   [i for i, (m, _) in enumerate(rank_opus, 1)
                              if m == "claude-haiku-4-5-20251001"][0],
    }
    Path("data/haiku_rejudge_report.json").write_text(json.dumps(report, indent=2))
    print("\n", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
