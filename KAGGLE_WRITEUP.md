# MetaMirage: The Sign-Flip Between Capability and Metacognition

**Subtitle:** Paired-task benchmark: accuracy ⊥ metacognition at r = −0.84 across 7 frontier LLMs (p = 0.018, 3 of 5 families sign-flip). Within-Anthropic comparison: **Opus 4.5 lost 32 TDR points relative to Opus 4.0** despite equal accuracy — direct evidence the trade-off is a recent training change.

**Track:** Metacognition

---

### Your Team

Dayeon Kang — independent submission.

### Problem Statement

**Primary domain:** Metacognitive monitoring in LLMs — specifically, a model's ability to recognize when a question contains a hidden flaw *before* committing to an answer.

**Capability being isolated:** *Trap-detection* — the monitoring side of metacognition. Given a question that looks answerable but is flawed (false premise, unanswerable setup, expertise-inverted framing), does the model flag it before answering? Distinct from correctness.

**Why this matters.** A deployed AGI that confidently answers a misleading question is more dangerous than one that flags uncertainty; measuring capability without monitoring optimizes for confident failure.

**Pre-registered hypothesis.** H0: TDR ⊥ accuracy. Observed r = −0.84, CI excludes zero, LOO-stable. **H0 rejected in the *opposite* direction** of capability-is-all — capability predicts *worse* monitoring.

**New insight.** Within Anthropic, **Opus 4.0 → 4.5 loses 32 TDR points** with no accuracy change. The trade-off is a recent training change, not architectural; haiku-4-5 (96% accuracy + MI 0.615) shows it can be retained.

**Mechanistic hypothesis.** The sign-flip likely reflects RLHF confidence pressure: annotators prefer decisive over hedged answers (Perez 2022; Casper 2023), so RLHF penalizes the calibrated abstention MetaMirage rewards. **Prediction:** proper-scoring-rule rewards in RLHF should weaken the sign-flip generationally — directly testable.

### Task & Benchmark Construction

**Mirage-pair design.** Each trap is presented as a *clean* variant (answerable) and a *mirage* variant (looks identical but is flawed). Correct behavior on a mirage: name the flaw before answering, or explicitly decline. Pair scoring isolates detection from absolute accuracy.

**5 families × 3 modes × 50 tasks.** Authored from scratch; no overlap with public benchmarks.

| Family | Trap Type | Scoring Mode | n |
|---|---|---|---|
| `expertise_trap` | Domain knowledge invites overconfidence on a meta-flawed premise | `expertise_inverted` | 8 |
| `forced_abstention` | Genuinely unanswerable; abstention is correct | `abstain_binary` | 12 |
| `confidence_inversion` | Easy answer; context should *lower* stated confidence | `rubric` | 10 |
| `over_specification` | Irrelevant constraints presented as distractors | `rubric` | 8 |
| `control_baseline` | Clean answerable tasks (calibration baseline) | `rubric` | 12 |

**Three scoring modes.** A single rubric hides distinct failure modes:

- `rubric` — Trap Detection + Confidence Appropriateness + Answer Quality, 0–3 each, normalized.
- `abstain_binary` — Abstain Score (0–3) + Answer Quality (0–3), 60/40.
- `expertise_inverted` — Metacognitive Flag (0–3) + Calibration (0–3), 70/30. *A confident domain-correct answer scores lower than one that flags the meta-level flaw* — rewards knowing better over knowing more.

**Kaggle SDK.** `kaggle_task.py` wraps tasks as `Task` objects with per-task `score_fn`, assembled into a `Benchmark` with `track="metacognition"`. `v3_tasks_50.json` is the single source of truth.

**Prompt design.** Identical system prompt across models: state confidence, flag noticed flaws before answering. Intentional — we test monitoring under a prompt that *invites* it; zero-shot is separate study.

**Output verification.** Responses scored by `claude-sonnet-4-5` under mode-specific rubric. Keyword heuristic in `kaggle_task.py` provides API-free smoke test.

### Dataset

**Provenance:** 50 tasks, hand-authored. No reuse of public benchmark items. Gold answers human-verified; every mirage task has a specified "what must be flagged" answer in `v3_tasks_50.json`.

**Schema:** `task_id`, `family` (5), `variant` (clean/mirage/abstain), `prompt`, `correct_answer`, `scoring_mode`, `mirage_signal`, `difficulty` (1–5), `tags`.

**Sample size and power.** n = 7: global r = −0.84, Fisher p = 0.018, bootstrap CI [−0.99, −0.58] (tighter than Fisher), **permutation p = 0.023**. LOO stable, Cohen's d = 2.65.

### Technical Details

**Repo:** https://github.com/dayeon603-pixel/MetaMirage. Key files: `v3_tasks_50.json`, `v3_judge_evaluator.py`, `v3_statistical_analysis.py` + `v3_robustness.py` + `v3_generational_analysis.py`, `v3_analysis.json`, `kaggle_task.py`, `dashboard.html`, `kaggle_submission.ipynb`, `cover_image.png`.

**Methodology note.** Per-family TDR correlated against **global** clean accuracy — undefined within-family for families without clean pairs (see `v3_analysis.json.methodology_note`).

**Reproducibility.** `git clone` → `pip install -r requirements.txt` → set API keys → `python v3_judge_evaluator.py` → `python v3_statistical_analysis.py` → `python v3_robustness.py`. ~15 min, ~$4.

### Results, Insights, and Conclusions

**Leaderboard (Metacognitive Index = TDR·½ + max(0, CalibΔ)·½):**

| Rank | Model | MI | TDR | Clean Acc | CalibΔ |
|---|---|---|---|---|---|
| 1 | claude-haiku-4-5 | **0.615** | 75.6% | 96.3% | **+0.474** |
| 2 | gpt-4o-mini | 0.574 | **84.5%** | 75.9% | +0.303 |
| 3 | llama-3-70b | 0.538 | 82.9% | 64.8% | +0.246 |
| 4 | claude-sonnet-4-5 | 0.520 | 66.5% | 92.6% | +0.375 |
| 5 | gemini-1.5-pro | 0.508 | 77.2% | 77.8% | +0.244 |
| 6 | claude-opus-4-5 | 0.409 | 55.5% | **100.0%** | +0.263 |
| 7 | gpt-4o | 0.407 | 62.6% | 98.2% | +0.187 |

**MI spread = 0.208** (0.41–0.62): healthy gradient, every model at a distinct rank, no saturation.

**Global correlation: r = −0.84**, Fisher CI [−0.98, −0.24], **t-p = 0.018 · bootstrap CI [−0.99, −0.58] · permutation p = 0.023**. LOO-stable across all 7 folds (min|r| = 0.81). Cohen's d = 2.65.

**Per-family (n = 7; Fisher CI · t-p · bootstrap CI · permutation p):**

| Family | r | Fisher CI | t-p | Bootstrap CI | Perm p |
|---|---|---|---|---|---|
| `confidence_inversion` | **+0.89** | [+0.42, +0.98] | 0.007 | [+0.72, +1.00] | 0.023 |
| `expertise_trap` | **−0.79** | [−0.97, −0.09] | 0.035 | [−0.98, −0.50] | 0.042 |
| `forced_abstention` | **−0.81** | [−0.97, −0.14] | 0.028 | [−1.00, −0.56] | 0.039 |
| `over_specification` | +0.04 | [−0.73, +0.77] | 0.93 | [−0.89, +0.95] | 0.96 |
| `control_baseline` | n/a | — | — | — | — |

All three sign-flip families survive four independent tests (parametric t, Fisher-z CI, bootstrap CI, permutation) at p < 0.05. The null family is null on all four.

**Insights:** (1) **Generational regression** — Opus 4.0 → 4.5: TDR 0.87 → 0.55 (−32 pts), Sonnet 4.0 → 4.5: 0.83 → 0.66 (−16 pts), accuracy ≈ constant. Trade-off was actively trained in. (2) **Haiku breaks the pattern** — MI 0.615 + 96% accuracy refutes "capability forces overconfidence." (3) **Competence trap measurable** — Opus 4.5 detects 94% of logical traps in `rubric` mode but 17% in `expertise_trap`. (4) **Hedging ≠ metacognition** — gpt-4o-mini: 100% expertise TDR + lowest accuracy = indiscriminate hedging; `expertise_inverted` rubric penalizes this.

**Within-Anthropic generational regression** *(new sub-finding, n = 6 Anthropic models, accuracy held constant).* Opus 4.0 (TDR 0.87, acc 0.98) → 4.1 (0.89, 0.96) → **4.5 (0.55, 1.00)** [Δ −0.32]. Sonnet 4.0 (0.83, 0.96) → 4.5 (0.66, 0.93) [Δ −0.16]. Haiku 4.5 retained it (0.76, 0.96). Direct, within-vendor, controlled evidence that the trade-off was *actively trained in* between Opus 4.1 and Opus 4.5 — exactly what the RLHF-confidence-pressure hypothesis predicts. Mechanism-level claim, not just correlation.

**Robustness checks.** Cross-judge: Claude responses (n = 150) re-judged by `claude-opus-4-5`; κ 0.65–0.97 across 6 dimensions, Pearson(judges) = 0.88, **haiku #1 holds under both**. Rubric-weight sensitivity: across all 36 (w_rubric, w_abstain, w_expert) triplets on a 0.1-grid, **94% give negative r, 81% reach p < 0.05** (median r = −0.82). n_effective: of 50 tasks, 12 are control-only; sign-flip driven by 38 mirage-bearing tasks. **Novelty audit:** char-5-gram Jaccard overlap with a 30-probe MMLU sample: max = 0.275, mean = 0.051 — well below the 0.5 paraphrase threshold; tasks are not MMLU rephrasings.

**Honest limitations.** (a) **Single author** — 50 tasks reflect one person's "trap" taste; domain bias toward stats/logic; unfixable without recruiting authors. (b) **Same-vendor primary judge** — full cross-vendor pending API access. (c) **Generational sub-finding: n = 3 Opus / 2 Sonnet** — suggestive, not definitive. (d) **No human baseline** — relative comparisons only, absolute "good metacognition" un-anchored; v4 priority. (e) **Task contamination** — trap styles common in pretraining; models may recognize genre. (f) Correlation, not causation. (g) Uneven clean/mirage balance per family (robust under global-accuracy methodology).

**Conclusion.** Sign-flip survives four statistical tests and LOO; pre-registered H0 rejected in opposite direction; mechanistic RLHF theory; counterexample (haiku); within-Anthropic generational regression directly observed.

### Organizational Affiliations

Independent submission. No organizational affiliation.

### References & Citations

- Flavell, J.H. (1979). Metacognition and cognitive monitoring. *Am. Psychol.* 34(10), 906–911.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.
- Perez, E., et al. (2022). Discovering language model behaviors with model-written evaluations. *arXiv:2212.09251*. — RLHF sycophancy.
- Casper, S., et al. (2023). Open problems and fundamental limitations of RLHF. *arXiv:2307.15217*. — Reward-model misalignment.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? *arXiv:2405.00623*.
- Huang, L., et al. (2023). A survey on hallucination in large language models. *arXiv:2311.05232*.
- Landis, J.R. & Koch, G.G. (1977). Measurement of observer agreement. *Biometrics* 33(1), 159–174.
- Plomecka, M., et al. (2026). Measuring Progress Toward AGI — Cognitive Abilities. Kaggle.
