"""
CognitiveMirage v3 — Surgical regeneration of family correlations.

Raw per-task judge records are not committed; only aggregated profiles live in
v3_analysis.json. This script patches the family_correlations and LOO blocks
using the corrected methodology (family TDR vs GLOBAL aq_clean), preserving
all other fields (global_correlation, profiles, effect size, etc.).

Output: overwrites v3_analysis.json with corrected family stats.
Old file is backed up to v3_analysis.json.bak.
"""
from __future__ import annotations
import json, math, shutil
from pathlib import Path

ANALYSIS = Path(__file__).parent / "v3_analysis.json"
BACKUP   = Path(__file__).parent / "v3_analysis.json.bak"


def pearson(xs, ys):
    if len(xs) < 3: return 0.0, 1.0
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    if den == 0: return 0.0, 1.0
    r = num/den
    n = len(xs)
    t = r*math.sqrt(n-2)/math.sqrt(max(1e-9, 1-r**2))
    p = 2*(1 - 0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return round(r,4), round(p,4)


def fisher_ci(r, n, alpha=0.05):
    if r is None or n < 4: return [None, None]
    r = max(min(r, 0.999999), -0.999999)
    z = 0.5*math.log((1+r)/(1-r))
    se = 1/math.sqrt(n-3)
    zcrit = 1.959964
    return [round(math.tanh(z - zcrit*se), 4),
            round(math.tanh(z + zcrit*se), 4)]


def main():
    a = json.loads(ANALYSIS.read_text())
    shutil.copy(ANALYSIS, BACKUP)

    profiles = a["profiles"]
    models = list(profiles.keys())
    families = sorted({f for p in profiles.values() for f in p["family"]})

    global_accs = [profiles[m]["aq_clean"] for m in models]

    # === Corrected family correlations ===
    fam_corr_new = {}
    for fam in families:
        tdrs = [profiles[m]["family"][fam]["tdr"] for m in models]
        # Degenerate check: if all tdr values identical, correlation is undefined
        if len({round(t,6) for t in tdrs}) <= 1:
            fam_corr_new[fam] = {
                "r": None, "p": None, "n": len(models),
                "note": "degenerate (TDR constant across models — family has no discriminating variant for this axis)",
            }
            continue
        r, p = pearson(tdrs, global_accs)
        ci = fisher_ci(r, len(models))
        fam_corr_new[fam] = {"r": r, "p": p, "n": len(models), "ci95": ci}

    # === Corrected LOO stability ===
    loo_new = {}
    # Global (unchanged methodology but recomputed for consistency)
    def _loo(name, xfn, yfn):
        rs = []
        for held in models:
            rest = [m for m in models if m != held]
            if len(rest) < 3: continue
            xr = [xfn(m) for m in rest]
            yr = [yfn(m) for m in rest]
            if len({round(v,6) for v in xr}) <= 1:
                continue
            r, _ = pearson(xr, yr)
            rs.append(round(r, 4))
        if not rs: return None
        return {
            "loo_r_values": rs,
            "min_abs_r": round(min(abs(v) for v in rs), 4),
            "sign_stable": len({1 if v > 0 else -1 for v in rs}) == 1,
            "min_r": round(min(rs), 4),
            "max_r": round(max(rs), 4),
        }

    loo_new["global_tdr_vs_accuracy"] = _loo(
        "global",
        lambda m: profiles[m]["tdr_global"],
        lambda m: profiles[m]["aq_clean"],
    )
    for fam in families:
        res = _loo(
            fam,
            lambda m, f=fam: profiles[m]["family"][f]["tdr"],
            lambda m: profiles[m]["aq_clean"],
        )
        if res is not None:
            loo_new[f"family_{fam}_tdr_vs_accuracy"] = res
        else:
            loo_new[f"family_{fam}_tdr_vs_accuracy"] = {
                "note": "degenerate (TDR constant — no LOO signal)",
            }

    a["family_correlations"] = fam_corr_new
    a["loo_stability"] = loo_new
    a["methodology_note"] = (
        "Per-family TDR is correlated against GLOBAL aq_clean (the stable "
        "capability axis across all 50 tasks) rather than within-family "
        "clean_aq. This corrects a degenerate r=0 artifact in families that "
        "have no clean-pair tasks (over_specification, control_baseline). "
        "Global correlation and per-model profiles are unchanged."
    )

    ANALYSIS.write_text(json.dumps(a, indent=2))

    print("=== Corrected family correlations ===")
    for fam, v in sorted(fam_corr_new.items(), key=lambda x: (x[1]["r"] or 0)):
        r = v.get("r"); ci = v.get("ci95", [None,None]); p = v.get("p")
        if r is None:
            print(f"  {fam:22s} r=n/a    {v.get('note','')}")
        else:
            print(f"  {fam:22s} r={r:+.4f}  95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  p={p:.4f}")
    print(f"\nBackup: {BACKUP}")
    print(f"Wrote:  {ANALYSIS}")


if __name__ == "__main__":
    main()
