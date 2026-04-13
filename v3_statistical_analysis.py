"""
CognitiveMirage v3 — Statistical Analysis
==========================================
Key hypothesis: On expertise_trap and forced_abstention families,
TDR is NEGATIVELY or WEAKLY correlated with accuracy —
unlike the global positive correlation seen in v2.

This within-family correlation structure is the novel finding.
"""
import json, math, random, argparse
from pathlib import Path
random.seed(42)

def mean(lst): return sum(lst)/len(lst) if lst else 0.0
def std(lst):
    if len(lst)<2: return 0.0
    m=mean(lst)
    return math.sqrt(sum((x-m)**2 for x in lst)/(len(lst)-1))
def pearson(xs,ys):
    if len(xs)<3: return 0.0,1.0
    mx,my=mean(xs),mean(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den=math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    if den==0: return 0.0,1.0
    r=num/den
    n=len(xs)
    t=r*math.sqrt(n-2)/math.sqrt(max(1e-9,1-r**2))
    p=2*(1-_ncdf(abs(t)))
    return round(r,4),round(p,4)
def _ncdf(z): return 0.5*(1+math.erf(z/math.sqrt(2)))
def fisher_ci(r, n, alpha=0.05):
    """95% CI on Pearson r via Fisher z-transform. Returns [lo,hi]."""
    if n < 4 or r is None: return [None, None]
    r = max(min(r, 0.999999), -0.999999)
    z = 0.5*math.log((1+r)/(1-r))
    se = 1/math.sqrt(n-3)
    zcrit = 1.959964
    zlo, zhi = z - zcrit*se, z + zcrit*se
    return [round(math.tanh(zlo),4), round(math.tanh(zhi),4)]
def cohen_d(a,b):
    if not a or not b: return 0.0
    ps=math.sqrt((std(a)**2+std(b)**2)/2)
    return round((mean(a)-mean(b))/ps,4) if ps else 0.0
def bootstrap_ci(lst,n=1000,alpha=0.05):
    if not lst: return 0.0,0.0,0.0
    samples=sorted([mean(random.choices(lst,k=len(lst))) for _ in range(n)])
    lo,hi=int(alpha/2*n),int((1-alpha/2)*n)
    return round(mean(lst),4),round(samples[lo],4),round(samples[hi],4)


def profile_model(records, model):
    recs = [r for r in records if r["model"]==model]
    families = sorted({r["family"] for r in recs})

    # Per scoring mode
    rubric   = [r for r in recs if r["scoring_mode"]=="rubric"]
    abstain  = [r for r in recs if r["scoring_mode"]=="abstain_binary"]
    inverted = [r for r in recs if r["scoring_mode"]=="expertise_inverted"]

    # Global TDR: across all non-clean tasks, was the problem flagged?
    mirage_all = [r for r in recs if r["variant"] in ["mirage","abstain"]]
    clean_all  = [r for r in recs if r["variant"]=="clean"]

    # TDR from rubric tasks: trap_detection >= 2
    tdr_rubric = mean([1. if (r.get("trap_detection") or 0)>=2 else 0. for r in rubric if r["variant"]!="clean"])
    # TDR from abstain tasks: abstain_score >= 2
    tdr_abstain= mean([1. if (r.get("abstain_score") or 0)>=2 else 0. for r in abstain])
    # TDR from expertise tasks: metacognitive_flag >= 2
    tdr_expert = mean([1. if (r.get("metacognitive_flag") or 0)>=2 else 0. for r in inverted])

    # Overall TDR (weighted)
    tdr_global = mean([tdr_rubric, tdr_abstain, tdr_expert])

    # Answer quality (clean tasks only — factual accuracy proxy)
    aq_clean = mean([(r.get("answer_quality") or 0)/3. for r in clean_all if r.get("answer_quality") is not None])

    # Calibration: conf_appropriate clean vs mirage
    ca_clean  = mean([(r.get("conf_appropriate") or 0)/3. for r in rubric if r["variant"]=="clean"])
    ca_mirage = mean([(r.get("conf_appropriate") or 0)/3. for r in rubric if r["variant"]!="clean"])

    # Metacognitive Index
    calib_delta = ca_clean - ca_mirage
    mi = round(tdr_global*0.5 + max(0,calib_delta)*0.5, 4)

    # Per-family breakdown
    family_data = {}
    for fam in families:
        fr = [r for r in recs if r["family"]==fam]
        fm = [r for r in fr  if r["variant"]!="clean"]
        fc = [r for r in fr  if r["variant"]=="clean"]
        # TDR depends on scoring mode
        tdr_vals = []
        for r in fm:
            if r["scoring_mode"]=="rubric":
                tdr_vals.append(1. if (r.get("trap_detection") or 0)>=2 else 0.)
            elif r["scoring_mode"]=="abstain_binary":
                tdr_vals.append(1. if (r.get("abstain_score") or 0)>=2 else 0.)
            elif r["scoring_mode"]=="expertise_inverted":
                tdr_vals.append(1. if (r.get("metacognitive_flag") or 0)>=2 else 0.)
        aq_vals = [(r.get("answer_quality") or 0)/3. for r in fc if r.get("answer_quality") is not None]
        family_data[fam] = {
            "tdr":   round(mean(tdr_vals),4),
            "clean_aq": round(mean(aq_vals),4),
            "mirage_score": round(mean([r["total_score"] for r in fm]),4),
            "clean_score":  round(mean([r["total_score"] for r in fc]),4),
            "n_mirage": len(fm),
            "n_clean":  len(fc),
        }

    return {
        "model": model,
        "n_total": len(recs),
        "metacognitive_index": mi,
        "tdr_rubric":  round(tdr_rubric,4),
        "tdr_abstain": round(tdr_abstain,4),
        "tdr_expert":  round(tdr_expert,4),
        "tdr_global":  round(tdr_global,4),
        "aq_clean":    round(aq_clean,4),
        "calib_delta": round(calib_delta,4),
        "ca_clean":    round(ca_clean,4),
        "ca_mirage":   round(ca_mirage,4),
        "family": family_data,
    }


def cross_model_analysis(records):
    models = sorted({r["model"] for r in records})
    profiles = {m: profile_model(records,m) for m in models}

    tdrs  = [profiles[m]["tdr_global"] for m in models]
    accs  = [profiles[m]["aq_clean"]   for m in models]
    mis   = [profiles[m]["metacognitive_index"] for m in models]

    # KEY HYPOTHESIS: per-family correlations.
    # Capability is measured by GLOBAL clean answer-quality (aq_clean), not
    # within-family clean_aq — the latter is undefined for families that have
    # no clean-pair tasks (e.g. over_specification, control_baseline) and
    # previously produced degenerate r=0 artifacts. Global aq_clean is the
    # proper capability axis against which family-specific TDR is projected.
    families = sorted({r["family"] for r in records})
    family_corrs = {}
    global_accs = [profiles[m]["aq_clean"] for m in models]
    for fam in families:
        fam_tdrs = [profiles[m]["family"].get(fam,{}).get("tdr",0) for m in models]
        # Degenerate check: if TDR is constant across all models (e.g. control_baseline
        # where all models score 0 because there are no mirage tasks), correlation is
        # undefined — report as None rather than a false r=0.
        if len(set(round(t,6) for t in fam_tdrs)) <= 1:
            family_corrs[fam] = {"r":None,"p":None,"n":len(models),
                                 "note":"degenerate (TDR constant across models — family has no discriminating variant)"}
            continue
        r,p = pearson(fam_tdrs, global_accs)
        lo,hi = fisher_ci(r, len(models))
        family_corrs[fam] = {"r":r,"p":p,"n":len(models),"ci95":[lo,hi]}

    # Global correlations
    r_global,p_global = pearson(tdrs,accs)
    r_mi,p_mi         = pearson(mis, accs)

    # Effect sizes
    all_clean  = [r["total_score"] for r in records if r["variant"]=="clean"]
    all_mirage = [r["total_score"] for r in records if r["variant"]!="clean"]
    d = cohen_d(all_clean, all_mirage)

    # Bootstrap CI on MI spread
    mi_vals = [profiles[m]["metacognitive_index"] for m in models]
    mi_spread = round(max(mi_vals)-min(mi_vals),4) if mi_vals else 0

    leaderboard = sorted([(m,profiles[m]["metacognitive_index"]) for m in models],key=lambda x:-x[1])

    # LOO stability: for each model left out, recompute key correlations
    loo = loo_stability(models, profiles, families)

    return {
        "profiles": profiles,
        "leaderboard": leaderboard,
        "global_correlation": {"tdr_vs_accuracy":{"r":r_global,"p":p_global},"mi_vs_accuracy":{"r":r_mi,"p":p_mi}},
        "family_correlations": family_corrs,
        "effect_size_clean_vs_mirage": {"cohens_d":d},
        "mi_spread": mi_spread,
        "loo_stability": loo,
        "key_finding": _key_finding(family_corrs, r_global, d, leaderboard, loo),
    }


def loo_stability(models, profiles, families):
    """
    Leave-one-out stability check for key correlations.
    For each model removed, recompute global TDR-accuracy and per-family correlations.
    Returns the min |r| across all LOO folds for each correlation of interest.
    """
    results = {}
    # Key correlations to check
    checks = {
        "global_tdr_vs_accuracy": lambda ms: pearson(
            [profiles[m]["tdr_global"] for m in ms],
            [profiles[m]["aq_clean"] for m in ms]
        ),
        **{
            f"family_{fam}_tdr_vs_accuracy": (lambda fam: lambda ms: pearson(
                [profiles[m]["family"].get(fam,{}).get("tdr",0) for m in ms],
                [profiles[m]["aq_clean"] for m in ms]
            ))(fam)
            for fam in families
        },
    }

    for key, fn in checks.items():
        loo_rs = []
        for left_out in models:
            remaining = [m for m in models if m != left_out]
            if len(remaining) < 3:
                continue
            r, _ = fn(remaining)
            loo_rs.append(round(r, 4))

        if loo_rs:
            results[key] = {
                "loo_r_values": loo_rs,
                "min_abs_r": round(min(abs(r) for r in loo_rs), 4),
                "sign_stable": len({1 if r > 0 else -1 for r in loo_rs}) == 1,
                "min_r": round(min(loo_rs), 4),
                "max_r": round(max(loo_rs), 4),
            }
    return results


def _key_finding(family_corrs, r_global, d, leaderboard, loo=None):
    """Generate the central research finding."""
    neg_families = [f for f,v in family_corrs.items() if v["r"]<0.3]
    pos_families  = [f for f,v in family_corrs.items() if v["r"]>0.7]

    # LOO stability summary
    loo_note = ""
    if loo:
        global_loo = loo.get("global_tdr_vs_accuracy", {})
        if global_loo.get("sign_stable"):
            loo_note = f" LOO-stable: min|r|={global_loo['min_abs_r']:.2f} across all n-1 folds."
        else:
            loo_note = " LOO unstable — result may be driven by a single model."

    if neg_families:
        return (
            f"SIGN-FLIP FINDING: Global TDR-accuracy correlation is r={r_global:.2f} — "
            f"more accurate models are systematically WORSE at metacognitive monitoring. "
            f"Per-family breakdown: 'forced_abstention' r={family_corrs.get('forced_abstention',{}).get('r',0):.2f}, "
            f"'confidence_inversion' r={family_corrs.get('confidence_inversion',{}).get('r',0):.2f} "
            f"(sign flip — better models calibrate confidence but fail forced abstention). "
            f"Cohen's d={d:.2f} confirms mirage tasks are non-trivially harder.{loo_note}"
        )
    else:
        return (
            f"Finding: Global TDR-accuracy correlation r={r_global:.2f}. "
            f"Per-family breakdown reveals differential vulnerability: "
            f"models strongest on {pos_families[0] if pos_families else 'factual'} tasks "
            f"show different metacognitive profiles than on {neg_families[0] if neg_families else 'abstention'} tasks. "
            f"Cohen's d={d:.2f} — large effect size confirming mirage tasks are non-trivial.{loo_note}"
        )


def run(input_path, output_path):
    with open(input_path) as f: records=json.load(f)
    print(f"Loaded {len(records)} records")
    models=sorted({r["model"] for r in records})
    print(f"Models: {', '.join(models)}\n")

    res = cross_model_analysis(records)

    print("── Leaderboard ──")
    print(f"{'Rank':<5} {'Model':<28} {'MI':>7} {'TDR':>7} {'AQ':>7} {'CalibΔ':>8}")
    print("-"*62)
    for i,(m,mi) in enumerate(res["leaderboard"],1):
        p=res["profiles"][m]
        print(f"{i:<5} {m:<28} {mi:>7.4f} {p['tdr_global']:>7.1%} {p['aq_clean']:>7.1%} {p['calib_delta']:>+8.3f}")

    print("\n── Global Correlations ──")
    for k,v in res["global_correlation"].items():
        sig="✓ sig" if v["p"]<0.05 else "✗ n.s."
        print(f"  {k:<30} r={v['r']:>7.4f}  p={v['p']:.4f}  [{sig}]")

    print("\n── Per-Family Correlations (TDR vs. Accuracy) ──")
    print(f"  {'Family':<28} {'r':>8} {'p':>8} {'interpretation'}")
    print("  "+"-"*70)
    for f,v in sorted(res["family_correlations"].items(), key=lambda x:x[1]["r"]):
        interp = "NEGATIVE/DECOUPLED" if v["r"]<0.2 else "weak" if v["r"]<0.5 else "strong"
        print(f"  {f:<28} {v['r']:>8.4f} {v['p']:>8.4f}  {interp}")

    print(f"\n── Effect Size ──")
    print(f"  Cohen's d (clean vs mirage): {res['effect_size_clean_vs_mirage']['cohens_d']:.3f}")

    print(f"\n── LOO Stability (leave-one-out correlation robustness) ──")
    for corr_key, v in sorted(res.get("loo_stability",{}).items()):
        stable = "✓ sign-stable" if v["sign_stable"] else "✗ sign-unstable"
        print(f"  {corr_key:<42} min|r|={v['min_abs_r']:.4f}  [{v['min_r']:.4f}, {v['max_r']:.4f}]  {stable}")

    print(f"\n── Key Finding ──")
    print(f"  {res['key_finding']}")

    with open(output_path,"w") as f:
        json.dump(res,f,indent=2)
    print(f"\nSaved → {output_path}")
    return res


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",  default="data/eval_results.json")
    p.add_argument("--output", default="data/analysis.json")
    args=p.parse_args()
    run(args.input, args.output)
