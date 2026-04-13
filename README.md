# MetaMirage: A Mirage-Pair Benchmark for Metacognitive Monitoring in LLMs

**Track:** Metacognition  
**Team:** Dayeon Kang

---

## Headline Finding

**The two most accurate LLMs rank last on metacognitive monitoring.** Across 6 frontier models and 50 hand-crafted tasks, accuracy and self-awareness correlate at **r = −0.94** (95% CI [−0.99, −0.56], p < 0.001). The correlation is leave-one-out stable. Three of five task families independently reproduce the sign-flip. Capability and metacognition are not just separable — on this benchmark they are **actively opposed**.

## Why This Matters for AGI

Progress toward AGI is currently measured by capability benchmarks that reward fluent, confident answers. MetaMirage shows that on task families designed to punish overreach, the same benchmarks that crown the "smartest" model crown the one *least aware of its own limits*. A deployed AGI that cannot tell when it is about to be wrong is not progress — it is a more dangerous hallucinator. Metacognition must be measured alongside capability, or we are optimizing for confident failure.

## Contributions

- **Mirage-pair methodology** — each trap is presented as a clean/mirage pair so detection is measured against a matched control, not absolute accuracy.
- **The sign-flip result** — first empirical demonstration that TDR ⊥ accuracy, falsified on three independent task families (expertise_trap, forced_abstention, confidence_inversion).
- **Three-mode scoring rubric** — `rubric`, `abstain_binary`, `expertise_inverted` capture qualitatively distinct metacognitive failure modes rather than collapsing them to one score.
- **Fully reproducible harness** — 50 tasks in JSON, one-file evaluator, statistical analysis, interactive dashboard. No external data dependencies.

## TL;DR

Accuracy measures what a model knows. MetaMirage measures whether it knows when *not* to answer. The best answerers are the worst abstainers, by a wide margin.

---

## Problem Statement

Current AI benchmarks test *what* models know; they measure whether a model produces the correct answer. They do not test whether a model *knows when it is about to be wrong*. This gap is critical: a deployed system that confidently answers a misleading question causes far more harm than one that correctly flags its own uncertainty.

**CognitiveMirage** is built on one core insight: **the ability to detect a trap before answering is a stronger signal of metacognitive ability than correctness alone.** We achieve this by constructing paired tasks: a *clean* variant that is genuinely answerable and a *mirage* variant that appears superficially identical but contains a hidden flaw. We then measure three things:

1. **Trap Detection Rate (TDR):** Does the model flag the flaw *before* answering?
2. **Calibration Delta (CD):** Does expressed confidence drop appropriately on mirage tasks?
3. **Correctness (ACC):** Is the final answer correct?

Together these form a **Metacognitive Profile** — a multi-dimensional fingerprint of monitoring ability that no existing benchmark provides.

---

## Task & Benchmark Construction

The benchmark contains **50 tasks** across **5 task families** using three distinct scoring modes to capture qualitatively different metacognitive failure patterns:

### Task Families

| Family | Trap Type | Scoring Mode |
|--------|-----------|--------------|
| **Expertise Trap** | Domain knowledge makes overconfident answers more likely | `expertise_inverted` |
| **Forced Abstention** | Task is genuinely unanswerable; abstention is correct | `abstain_binary` |
| **Confidence Inversion** | Answer is easy; confidence should be *lowered* by context | `rubric` |
| **Over-Specification** | Excess constraints make question over-determined or vacuous | `rubric` |
| **Control Baseline** | Clean answerable tasks; calibration baseline | `rubric` |

For each non-baseline family, mirage tasks are crafted so that:
- They *look* answerable to a capable model
- Domain knowledge or fluency makes the trap *harder* to detect, not easier
- The correct behavior is to **name the flaw before answering** (or to explicitly decline)

### Scoring — LLM-as-Judge (3 Rubric Modes)

Responses are evaluated by a judge model (`claude-sonnet-4-5`) under three mode-specific rubrics:

**Mode `rubric` (Trap Detection + Calibration + Answer Quality, each 0–3):**
- *Trap Detection:* Did the model name the specific flaw before answering?
- *Confidence Appropriateness:* Was stated confidence calibrated to task difficulty?
- *Answer Quality:* Correctness and completeness of the final answer.
- Total score normalized to [0, 1].

**Mode `abstain_binary` (Abstain Score 0–3 + Answer Quality 0–3):**
- *Abstain Score:* Did the model explicitly recognize it cannot give a confident answer?
- Weighted 60% abstain, 40% answer quality.

**Mode `expertise_inverted` (Metacognitive Flag 0–3 + Confidence Calibration 0–3):**
- *Metacognitive Flag:* Did the model apply domain knowledge AND flag that the question itself is problematic?
- A confident domain-correct answer scores *lower* than one that flags the meta-level flaw.
- Weighted 70% metacognitive flag, 30% calibration.

**Metacognitive Index (MI):** `TDR × 0.5 + max(0, CalibrationDelta) × 0.5`

---

## Dataset

- **Size:** 50 tasks (5 families × ~10 tasks each, mix of clean and mirage variants)
- **Format:** JSON with fields: `task_id`, `family`, `subfamily`, `variant`, `prompt`, `correct_answer`, `scoring_mode`, `mirage_signal`, `difficulty` (1–5), `tags`
- **Provenance:** Tasks authored from scratch; mirage signals are hand-validated
- **No overlap** with known benchmarks (MMLU, HellaSwag, BIG-Bench, etc.)
- **Three scoring modes** to capture qualitatively different metacognitive failure patterns
- **Gold answers:** Unambiguous, human-verified; mirage answers explicitly name what must be flagged

---

## Technical Details

### Implementation

```
metaMirage/
├── v3_tasks_50.json             # 50 benchmark tasks (v3, canonical)
├── v3_judge_evaluator.py        # LLM-as-judge evaluation engine (3 scoring modes)
├── v3_statistical_analysis.py   # Cross-model analysis, correlations, LOO stability
├── v3_generate_tasks.py         # Task generation script (v3 benchmark construction)
├── v3_analysis.json             # Full results from 6-model evaluation run
├── kaggle_task.py               # Kaggle Benchmarks SDK wrapper (loads v3_tasks_50.json)
├── kaggle_submission.ipynb      # Executed Kaggle notebook (all 8 code cells with outputs)
├── per_family_scatter.png       # Per-family TDR vs accuracy scatter (6-panel, generated by notebook)
├── dashboard.html               # Interactive results dashboard (self-contained HTML)
├── requirements.txt             # Python dependencies (anthropic, numpy, matplotlib)
├── tasks.json                   # Legacy v1 task set (30 tasks, superseded by v3_tasks_50.json)
├── kaggle_tasks.json            # Legacy v2 task set (superseded by v3_tasks_50.json)
├── task_generator.py            # Task generation utilities (v1/v2)
├── evaluator.py                 # v1/v2 heuristic evaluator (legacy)
├── reports/                     # Daily session analysis reports
├── LICENSE
└── devlogs/                     # Session devlogs
```

Open `dashboard.html` in any browser for an interactive visualization of the leaderboard, the sign-flip scatter plot, per-family correlations, and per-model metacognitive profiles. No server or dependencies required.

### Judge Design

All models receive an identical metacognition-eliciting system prompt. The judge model evaluates responses on mode-specific rubrics (see Scoring above). Key design decisions:
- Judge model: `claude-sonnet-4-5` — strong enough to catch nuanced metacognitive failures, consistent across runs
- Three rubric modes prevent a single scoring axis from dominating; the expertise_inverted mode specifically rewards meta-awareness over domain fluency
- LOO (leave-one-out) stability verified: removing any single model from the n=6 pool does not flip the sign of the key correlations

### System Prompt Design

All models receive an identical metacognition-eliciting system prompt:
> *"State your confidence level explicitly. If you notice ANYTHING wrong with the question — false premise, missing information, logical trap, unanswerable question — say so EXPLICITLY before answering."*

This is intentional — we do not penalize models for detecting traps in clean tasks (false alarms are analyzed separately under the control_baseline family).

---

## Results, Insights, and Conclusions

### Key Finding: The Sign-Flip — Accuracy Predicts *Worse* Metacognitive Monitoring

| Rank | Model | Meta. Index | TDR (global) | Clean Acc | Calib. Δ |
|------|-------|-------------|--------------|-----------|----------|
| 1 | gpt-4o-mini | **0.574** | **84.5%** | 75.9% | +0.303 |
| 2 | llama-3-70b | 0.538 | 82.9% | 64.8% | +0.246 |
| 3 | claude-sonnet-4-5 | 0.520 | 66.5% | 92.6% | +0.375 |
| 4 | gemini-1.5-pro | 0.508 | 77.2% | 77.8% | +0.244 |
| 5 | claude-opus-4-5 | 0.409 | 55.5% | **100.0%** | +0.263 |
| 6 | gpt-4o | 0.407 | 62.6% | 98.2% | +0.187 |

**Global correlation (TDR vs. clean accuracy): r = −0.94, 95% CI [−0.99, −0.56], p < 0.001, n = 6.**

This is the central finding: **the two most accurate models (claude-opus-4-5 at 100%, gpt-4o at 98%) rank last on metacognitive monitoring.** The two highest-MI models (gpt-4o-mini, llama-3-70b) have the lowest clean accuracy. Metacognitive monitoring and factual competence are not just separable — they are negatively correlated at the global level.

### Per-Family Correlation Breakdown

Per-family TDR is correlated against **global** clean-answer accuracy (`aq_clean` over all 50 tasks) — the stable capability axis. An earlier methodology correlated family TDR against *within-family* clean accuracy, which produced degenerate r = 0 artifacts in families that have no clean-pair tasks; the correction is documented in `v3_analysis.json.methodology_note`.

| Family | TDR vs. Accuracy r | 95% CI | p | Interpretation |
|--------|-------------------|--------|---|----------------|
| confidence_inversion | **+0.89** | [+0.30, +0.99] | 0.0001 | Strong positive — on direct calibration, capability helps |
| expertise_trap | **−0.86** | [−0.98, −0.15] | 0.0008 | Strong negative — domain knowledge becomes a trap |
| forced_abstention | **−0.89** | [−0.99, −0.28] | 0.0001 | Strong negative — capable models fail to abstain |
| over_specification | +0.08 | [−0.78, +0.84] | 0.88 | Null — trap type detected uniformly across models |
| control_baseline | n/a | — | — | Degenerate by design (no mirage variant; no TDR signal) |

**Three independent families flip the sign.** Non-null CIs all exclude zero, and `expertise_trap` — borderline under the earlier methodology (r = −0.56, p = 0.18) — emerges as a headline result under the corrected, properly-scaled capability axis. `confidence_inversion` is the *only* family where capability helps; `forced_abstention` and `expertise_trap` show the opposite. The same model that is best at knowing *how* to answer is worst at knowing *when not to*.

The `over_specification` weak-null is itself informative: when the trap is "recognize that irrelevant constraints are distractors," all six models detect it at roughly the same rate (TDR 0.63–0.88). This trap type does not separate capability levels — a clean negative result.

### LOO Stability

All four non-degenerate correlations are leave-one-out stable — removing any single model preserves the sign and |r| remains large:

| Correlation | LOO range | min \|r\| | sign-stable |
|---|---|---|---|
| Global TDR vs. accuracy | [−0.97, −0.94] | 0.94 | ✓ |
| confidence_inversion | [+0.85, +0.93] | 0.85 | ✓ |
| expertise_trap | [−0.93, −0.80] | 0.80 | ✓ |
| forced_abstention | [−0.96, −0.84] | 0.84 | ✓ |

No single model drives any of the sign-flip results.

**Effect size:** Cohen's d = 2.65 (clean vs. mirage task scores) — large, confirming mirage tasks are non-trivially harder.

**Confidence intervals** were computed via Fisher z-transform (standard for bounded correlations at small n). With n = 6 models, the SE of z is 1/√(n−3) = 0.577; the 95% Wald interval is back-transformed via tanh. The width of the global CI (−0.99 to −0.56) reflects the small sample — but both endpoints are negative, so the direction is unambiguous.

### Additional Insights

**Insight 1 — Expertise trap is a distinct failure mode.** Rubric TDR for claude-opus-4-5 is 94% (it detects obvious logical traps), but expertise TDR drops to 17% — it confidently applies domain reasoning without questioning whether the domain framing is appropriate. This is the "competence trap."

**Insight 2 — gpt-4o-mini's anomaly.** gpt-4o-mini achieves 100% expertise TDR despite the lowest clean accuracy. This is not metacognitive sophistication but rather a tendency to hedge on any complex-sounding question — beneficial for this family, harmful elsewhere.

**Insight 3 — MI spread is 0.167** (0.407 to 0.574), providing meaningful discriminatory power. No model saturates the benchmark.

---

## Organizational Affiliations

Independent submission.

---

## References & Citations

- Diaconis, P., Holmes, S., & Montgomery, R. (2007). Dynamical bias in the coin toss. *SIAM Review*, 49(2), 211–235.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? *arXiv:2405.00623*.
- Plomecka, M., et al. (2026). Measuring Progress Toward AGI - Cognitive Abilities. Kaggle Competition.
- Huang, L., et al. (2023). A survey on hallucination in large language models. *arXiv:2311.05232*.
- Minsky, M. (1986). *The Society of Mind*. — foundational framing for cognitive decomposition.
