# MetaMirage: The Sign-Flip Between Capability and Metacognition

**Subtitle:** Paired-task benchmark: accuracy and metacognition correlate at r = −0.84 across 7 frontier LLMs (p = 0.018; 3 of 5 families sign-flip at p < 0.04). The dissociation is *trainable out* — claude-haiku-4-5 tops the leaderboard at 96% accuracy.

**Track:** Metacognition

---

### Your Team

Dayeon Kang — independent submission.

### Problem Statement

**Primary domain:** Metacognitive monitoring in LLMs — specifically, a model's ability to recognize when a question contains a hidden flaw *before* committing to an answer.

**Capability being isolated:** *Trap-detection* — the monitoring side of metacognition. Given a question that looks answerable but is flawed (false premise, unanswerable setup, expertise-inverted framing), does the model flag it before answering? Dissociated from correctness.

**Why this matters.** A deployed AGI that confidently answers a misleading question is more dangerous than one that flags uncertainty. Measuring capability without monitoring optimizes for confident failure.

**The new insight.** Capability and monitoring are negatively correlated across 7 frontier models (r = −0.84, p = 0.018) — but `claude-haiku-4-5` partially breaks the pattern (96.3% accuracy + MI = 0.615), showing the trade-off is **trainable, not architectural**.

### Task & Benchmark Construction

**Mirage-pair design.** For each trap, the benchmark presents a *clean* variant (genuinely answerable) and a *mirage* variant that looks superficially identical but contains a hidden flaw. Correct behavior on a mirage is to **name the flaw before answering**, or — for forced-abstention tasks — to explicitly decline. Scoring paired variants isolates detection from baseline accuracy.

**5 families × 3 scoring modes × 50 tasks.** All tasks authored from scratch; no overlap with MMLU, BIG-Bench, HellaSwag, or other public suites.

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

**Kaggle SDK.** `kaggle_task.py` wraps the 50 tasks as `Task` objects with per-task `score_fn` and metadata, assembled into a `Benchmark` with `track="metacognition"`. `v3_tasks_50.json` is the single source of truth for the SDK, the evaluator, and the analysis.

**Prompt robustness.** All models receive an identical system prompt that instructs them to state confidence, flag any noticed flaw *before* answering, then answer. Intentional — we test monitoring under a prompt that *invites* it; zero-shot monitoring is a separate study.

**Output verification robustness.** Responses are scored by `claude-sonnet-4-5` as judge under the mode-specific rubric. Judge prompts are versioned and open. A frozen-keyword heuristic (`kaggle_task.py:evaluate_response`) reproduces the judge's key signals for API-free smoke tests.

### Dataset

**Provenance:** 50 tasks, all authored from scratch for this submission. No scraping, no reuse of public benchmark items. Gold answers are unambiguous and human-verified; every mirage task has a single, specified "what must be flagged" answer documented in `v3_tasks_50.json`.

**Schema** (`v3_tasks_50.json`): `task_id` (10-char hex), `family` (one of 5), `variant` (`clean`/`mirage`/`abstain`), `prompt` (string), `correct_answer` (gold or, for mirage, the specific flaw to flag), `scoring_mode` (`rubric`/`abstain_binary`/`expertise_inverted`), `mirage_signal` (what must be flagged; mirage only), `difficulty` (1–5), `tags` (list[string]).

**Sample size and statistical power.** At n = 7 models the global correlation (r = −0.84) clears p < 0.05 with a Fisher CI excluding zero, LOO stability across all 7 folds, and Cohen's d = 2.65 between clean and mirage tasks. Per-family TDR spreads range 0.25–0.83 — every non-null family discriminates at least half the model pool.

### Technical Details

**Repo:** https://github.com/dayeon603-pixel/MetaMirage — single source of truth, everything traceable.

```
v3_tasks_50.json              50 benchmark tasks (canonical)
v3_judge_evaluator.py         LLM-as-judge evaluation engine (3 scoring modes)
v3_statistical_analysis.py    Cross-model stats, LOO, Fisher CIs, effect sizes
v3_regenerate_family_stats.py Surgical regenerator (corrected methodology patch)
v3_analysis.json              Full results from the 7-model evaluation
kaggle_task.py                Kaggle Benchmarks SDK wrapper (identical task set)
kaggle_submission.ipynb       Executed public notebook
dashboard.html                Self-contained interactive results dashboard
cover_image.png               Cover
requirements.txt              anthropic, numpy, matplotlib
```

**Methodology note.** Per-family TDR is correlated against **global** clean accuracy — the stable capability axis over all 50 tasks. An earlier draft used within-family clean accuracy, which was undefined for families without clean-pair tasks (documented in `v3_analysis.json.methodology_note`).

**Reproducibility.** `git clone` → `pip install -r requirements.txt` → set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` → `python v3_judge_evaluator.py --models <6 models> --tasks v3_tasks_50.json --output data/eval_results.json` → `python v3_statistical_analysis.py --input data/eval_results.json --output v3_analysis.json`. Runtime ~15 min, cost ~$3.

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

**MI spread = 0.208** (0.41–0.62): healthy gradient, no saturation, every model at a distinct rank. Clean accuracy ranges 64.8%–100%.

**Global correlation: r = −0.84**, 95% CI [−0.98, −0.24], **p = 0.018** (Student's t, df = 5), LOO-stable (all 7 folds |r| ≥ 0.81). Cohen's d = 2.65.

**Per-family correlations (n = 7, Student's t df = 5, Fisher-z 95% CI):**

| Family | r | 95% CI | p | LOO min \|r\| |
|---|---|---|---|---|
| `confidence_inversion` | **+0.89** | [+0.42, +0.98] | 0.007 | 0.86 ✓ |
| `expertise_trap` | **−0.79** | [−0.97, −0.09] | 0.035 | 0.74 ✓ |
| `forced_abstention` | **−0.81** | [−0.97, −0.14] | 0.028 | 0.76 ✓ |
| `over_specification` | +0.04 | [−0.73, +0.77] | 0.93 | n/a (null) |
| `control_baseline` | n/a | — | — | degenerate by design |

**Four insights:**

1. **Haiku breaks the sign-flip — the trade-off is trainable.** `claude-haiku-4-5` tops MI at 0.615 with 96.3% accuracy, sitting above all 6 larger models. A small, recent model with strong accuracy *and* strong monitoring is a direct counterexample to "capability forces overconfidence." The benchmark shifts from diagnostic to prescriptive: whatever training protocol produced haiku is a candidate solution.

2. **Where the flip persists, it is family-coherent.** `confidence_inversion` rewards capability; `forced_abstention` + `expertise_trap` punish it. The best *answerer* is the worst *abstainer* on two of three non-null families.

3. **The competence trap is measurable.** `claude-opus-4-5` detects 94% of logical traps in `rubric` mode but 17% in `expertise_trap` — it catches explicit flaws but confidently applies domain reasoning without questioning the framing itself.

4. **Hedging ≠ metacognition.** `gpt-4o-mini` posts 100% `expertise_trap` TDR alongside the lowest clean accuracy — it hedges indiscriminately. The `expertise_inverted` rubric penalizes undifferentiated uncertainty and separates genuine monitoring from defensive hedging.

**Cross-judge validation.** Claude-model responses (n = 150) re-judged by `claude-opus-4-5` alongside primary `claude-sonnet-4-5`. Weighted κ ranges 0.65–0.97 across all 6 rubric dimensions (substantial to near-perfect; Landis-Koch). Total-score Pearson between judges = 0.88. **Haiku's #1 ranking holds under both judges** (MI = 0.615 under sonnet; MI = 0.680 under opus) — the self-preference attack fails. Cross-vendor (GPT-4o) pending.

**Limitations.** (1) n = 7 models — CIs are wide but all non-null CIs exclude zero and every sign-flip is LOO-stable. (2) Primary judge `claude-sonnet-4-5`; intra-vendor cross-judge κ ≥ 0.65 on the Anthropic subset; full cross-vendor pending. (3) Single author. (4) **Uneven clean/mirage balance per family** — `confidence_inversion` is 9 mirage + 1 clean, `over_specification` is mirage-only, `expertise_inverted` mode spans only 6 tasks. Correlating family TDR against global accuracy is robust to this asymmetry. (5) Correlation, not causation.

**Conclusion.** MetaMirage finds an effect that survives LOO across 7 models, excludes zero on three independent families, and — via haiku — shows the sign-flip is trainable. Exactly the signal an AGI-progress benchmark must surface.

### Organizational Affiliations

Independent submission. No organizational affiliation.

### References & Citations

- Kadavath, S., et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? *arXiv:2405.00623*.
- Huang, L., et al. (2023). A survey on hallucination in large language models. *arXiv:2311.05232*.
- Flavell, J.H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906–911. — Foundational framing of monitoring as distinct from cognition.
- Plomecka, M., et al. (2026). Measuring Progress Toward AGI — Cognitive Abilities. Kaggle competition.
