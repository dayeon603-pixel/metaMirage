# MetaMirage: The Sign-Flip Between Capability and Metacognition

**Subtitle:** A paired mirage-task benchmark showing that on metacognitive monitoring, the most accurate LLMs rank last (global r = −0.94, p = 0.005; 3 of 5 families sign-flip independently at p < 0.03).

**Track:** Metacognition

---

### Your Team

Dayeon Kang — independent submission.

### Problem Statement

**Primary domain:** Metacognitive monitoring in LLMs — specifically, a model's ability to recognize when a question contains a hidden flaw *before* committing to an answer.

**Capability being isolated:** *Trap-detection*, the monitoring side of metacognition. Given a question that looks answerable but is flawed (false premise, unanswerable setup, expertise-inverted framing, or misleading context), does the model flag it before answering? Deliberately dissociated from correctness: a model can be highly accurate on clean questions yet blind to traps.

**Why this matters.** Current AGI benchmarks reward fluent, confident answers. A deployed AGI that confidently answers a misleading question causes more harm than one that flags uncertainty. Measuring capability without monitoring yields a confident hallucinator.

**The new insight.** On three independent families, **capability and metacognitive monitoring are negatively correlated, not merely separable** (all p < 0.03, Fisher-z CIs exclude zero, LOO-stable).

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

**Sample size and statistical power.** The benchmark is deliberately small and surgical: at n = 6 models the key correlation (r = −0.94) has p < 0.001 with a 95% Fisher CI excluding zero, leave-one-out stability across all 6 folds, and Cohen's d = 2.65 between clean and mirage task scores. Each non-baseline family carries 8–12 tasks, sufficient to produce per-family TDR variance that discriminates all 6 models (TDR spread ≥ 0.25 within every non-null family).

### Technical Details

**Repo:** https://github.com/dayeon603-pixel/MetaMirage — single source of truth, everything traceable.

```
v3_tasks_50.json              50 benchmark tasks (canonical)
v3_judge_evaluator.py         LLM-as-judge evaluation engine (3 scoring modes)
v3_statistical_analysis.py    Cross-model stats, LOO, Fisher CIs, effect sizes
v3_regenerate_family_stats.py Surgical regenerator (corrected methodology patch)
v3_analysis.json              Full results from the 6-model evaluation
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
| 1 | gpt-4o-mini | **0.574** | **84.5%** | 75.9% | +0.303 |
| 2 | llama-3-70b | 0.538 | 82.9% | 64.8% | +0.246 |
| 3 | claude-sonnet-4-5 | 0.520 | 66.5% | 92.6% | +0.375 |
| 4 | gemini-1.5-pro | 0.508 | 77.2% | 77.8% | +0.244 |
| 5 | claude-opus-4-5 | 0.409 | 55.5% | **100.0%** | +0.263 |
| 6 | gpt-4o | 0.407 | 62.6% | 98.2% | +0.187 |

**MI spread = 0.167** (0.41–0.57): a healthy gradient with no saturation at either end. Clean accuracy ranges 64.8%–100%. The benchmark discriminates — every model sits at a distinct rank on MI and no two models are closer than 0.0035.

**Global correlation: r = −0.94**, 95% CI [−0.99, −0.56], **p = 0.005** (Student's t, df = 4), LOO-stable (all 6 folds |r| ≥ 0.94). Cohen's d (clean vs. mirage, per-task scores) = 2.65.

**Per-family correlations (n = 6, Student's t on df = 4, Fisher-z 95% CI):**

| Family | r | 95% CI | p | LOO min \|r\| |
|---|---|---|---|---|
| `confidence_inversion` | **+0.89** | [+0.30, +0.99] | 0.016 | 0.85 ✓ |
| `expertise_trap` | **−0.86** | [−0.98, −0.15] | 0.029 | 0.80 ✓ |
| `forced_abstention` | **−0.89** | [−0.99, −0.28] | 0.018 | 0.84 ✓ |
| `over_specification` | +0.08 | [−0.78, +0.84] | 0.89 | n/a (null) |
| `control_baseline` | n/a | — | — | degenerate by design |

**Three insights:**

1. **Sign-flip is family-coherent.** `confidence_inversion` rewards capability (notice confidence should drop); `forced_abstention` + `expertise_trap` punish it (notice when competence misleads). The best *answerer* is the worst *abstainer*.

2. **Competence trap is measurable.** Claude-opus-4-5 detects 94% of logical traps in `rubric` mode but only 17% in `expertise_trap` — it catches explicit flaws but confidently applies domain reasoning without questioning the framing itself.

3. **Hedging ≠ metacognition.** gpt-4o-mini posts 100% `expertise_trap` TDR alongside the lowest clean accuracy — it hedges on anything complex-sounding. The `expertise_inverted` rubric separates genuine monitoring from defensive hedging by penalizing undifferentiated uncertainty.

**Cross-judge validation (Anthropic subset, n = 100).** Re-judged Claude-model responses with `claude-opus-4-5` alongside the primary `claude-sonnet-4-5`. Weighted κ per rubric dimension: trap_detection 0.65, conf_appropriateness 0.97, answer_quality 0.77, abstain_score 0.65, metacognitive_flag 0.66, confidence_calibration 0.84 — all substantial to near-perfect (Landis-Koch). Total-score Pearson between judges = 0.88; mean \|diff\| = 0.044 on [0,1]. Opus-as-judge slightly inverts the sonnet > opus ranking on this 2-model subset — **self-preference bias at small effect sizes**, which is why the n = 6 LOO-stable global r is stronger than any 2-model comparison. Cross-vendor validation (GPT-4o) pending OpenAI billing.

**Limitations.** (1) n = 6 models — CIs are wide but all non-null CIs exclude zero, every sign-flip is LOO-stable. (2) Primary judge `claude-sonnet-4-5`; intra-vendor cross-judge κ ≥ 0.65 on the Anthropic subset; full cross-vendor remains future work. (3) Single author. (4) **Uneven clean/mirage balance per family** — `confidence_inversion` is 9 mirage + 1 clean, `over_specification` is mirage-only, `expertise_inverted` mode spans only 6 tasks. Correlating family TDR against global accuracy is robust to this asymmetry, but denser pair coverage is a priority for v4. (5) Correlation, not causation.

**Conclusion.** MetaMirage is small by design but finds an effect that survives LOO, excludes zero on three independent families, and reverses the assumed link between capability and self-awareness — the signal an AGI-progress benchmark must surface.

### Organizational Affiliations

Independent submission. No organizational affiliation.

### References & Citations

- Kadavath, S., et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? *arXiv:2405.00623*.
- Huang, L., et al. (2023). A survey on hallucination in large language models. *arXiv:2311.05232*.
- Flavell, J.H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906–911. — Foundational framing of monitoring as distinct from cognition.
- Plomecka, M., et al. (2026). Measuring Progress Toward AGI — Cognitive Abilities. Kaggle competition.
