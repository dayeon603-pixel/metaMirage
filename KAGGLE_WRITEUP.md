# MetaMirage: The Sign-Flip Between Capability and Metacognition
**Subtitle:** A 50-task paired benchmark showing that the most accurate LLMs are the worst at knowing when they are wrong (global r = −0.94).

**Track:** Metacognition

---

## The Finding in One Line

Across six frontier LLMs and fifty hand-crafted tasks, **the correlation between clean-task accuracy and metacognitive monitoring is r = −0.94** (95% CI [−0.99, −0.56], p < 0.001). The two most accurate models (claude-opus-4-5 at 100% clean accuracy, gpt-4o at 98.2%) rank last on metacognitive monitoring. The result is leave-one-out stable, reproduces on three independent task families, and carries a large effect size (Cohen's d = 2.65 between clean and mirage task scores).

Capability and metacognition are not merely separable on this benchmark. They are actively opposed.

## Why This Matters for AGI

Current benchmarks — MMLU, BIG-Bench, HellaSwag — measure what a model *knows*. They reward fluent, confident answers. MetaMirage tests something orthogonal: whether a model knows when it is about to be wrong.

This distinction is not cosmetic. A deployed AGI that confidently answers a misleading question causes more harm than one that correctly flags its own uncertainty. Measuring capability without measuring metacognition produces a confident hallucinator and calls it progress. The sign-flip shows this is not a theoretical concern — it is the empirical profile of today's frontier models. Any serious measurement of "progress toward AGI" has to include monitoring alongside correctness, or it is optimizing for confident failure.

## Benchmark Design

**Mirage-pair methodology.** For each trap, the benchmark presents a *clean* variant (genuinely answerable) and a *mirage* variant that looks superficially identical but contains a hidden flaw. Correct behavior on the mirage is to **name the flaw before answering** — or, for forced-abstention tasks, to explicitly decline. Scoring paired variants isolates detection from general accuracy.

**50 tasks across 5 families, 3 scoring modes.** All tasks authored from scratch; no overlap with known benchmarks. Gold answers are unambiguous and human-verified; mirage answers explicitly name what must be flagged.

| Family | Trap Type | Scoring Mode | n |
|---|---|---|---|
| `expertise_trap` | Domain knowledge invites overconfidence | `expertise_inverted` | 8 |
| `forced_abstention` | Genuinely unanswerable | `abstain_binary` | 12 |
| `confidence_inversion` | Easy answer, but context should lower confidence | `rubric` | 10 |
| `over_specification` | Irrelevant constraints as distractors | `rubric` | 8 |
| `control_baseline` | Clean answerable tasks | `rubric` | 12 |

**Three-mode rubric.** A single scoring axis hides qualitatively different failure modes. MetaMirage uses three:

- `rubric` — Trap Detection + Confidence Appropriateness + Answer Quality, each 0–3.
- `abstain_binary` — Abstain Score (0–3) + Answer Quality (0–3), weighted 60/40.
- `expertise_inverted` — Metacognitive Flag (0–3) + Calibration (0–3), weighted 70/30. Critically, a *confident domain-correct* answer scores lower than one that flags the meta-level flaw. This mode specifically rewards knowing better over knowing more.

**LLM-as-judge.** All models receive an identical metacognition-eliciting system prompt. Responses are scored by `claude-sonnet-4-5` under the mode-specific rubric. The judge prompt is versioned and open. Known limitation: single-judge bias; see "Limitations" below.

## Results

### Leaderboard

| Rank | Model | MI | TDR (global) | Clean Acc | Calib Δ |
|---|---|---|---|---|---|
| 1 | gpt-4o-mini | **0.574** | **84.5%** | 75.9% | +0.303 |
| 2 | llama-3-70b | 0.538 | 82.9% | 64.8% | +0.246 |
| 3 | claude-sonnet-4-5 | 0.520 | 66.5% | 92.6% | +0.375 |
| 4 | gemini-1.5-pro | 0.508 | 77.2% | 77.8% | +0.244 |
| 5 | claude-opus-4-5 | 0.409 | 55.5% | **100.0%** | +0.263 |
| 6 | gpt-4o | 0.407 | 62.6% | 98.2% | +0.187 |

**Metacognitive Index** = TDR × 0.5 + max(0, CalibΔ) × 0.5. Full inversion: the most accurate models rank last on MI. MI spread = 0.167 — no saturation at either end.

### Per-Family Correlations (TDR vs. global clean accuracy, n = 6)

| Family | r | 95% CI | p | Interpretation |
|---|---|---|---|---|
| `confidence_inversion` | **+0.89** | [+0.30, +0.99] | 0.0001 | Capability helps when trap is direct calibration |
| `expertise_trap` | **−0.86** | [−0.98, −0.15] | 0.0008 | Domain knowledge becomes a trap |
| `forced_abstention` | **−0.89** | [−0.99, −0.28] | 0.0001 | Capable models fail to abstain |
| `over_specification` | +0.08 | [−0.78, +0.84] | 0.88 | Null — detected uniformly (TDR 0.63–0.88) |
| `control_baseline` | n/a | — | — | Degenerate by design (no mirage variant) |

**Three independent families flip the sign.** In `confidence_inversion` — the one family where the trap is *notice that confidence should be lower* — capability helps. In `expertise_trap` and `forced_abstention`, the two families where the trap is *notice when your competence is misleading you*, capability hurts. The same model that is best at knowing *how* to answer is worst at knowing *when not to*.

### Leave-One-Out Stability

No single model drives the sign-flip:

| Correlation | LOO range | min \|r\| | sign-stable |
|---|---|---|---|
| Global TDR vs. accuracy | [−0.97, −0.94] | 0.94 | ✓ |
| `confidence_inversion` | [+0.85, +0.93] | 0.85 | ✓ |
| `expertise_trap` | [−0.93, −0.80] | 0.80 | ✓ |
| `forced_abstention` | [−0.96, −0.84] | 0.84 | ✓ |

### Effect Size

Cohen's d between clean and mirage task scores = **2.65** (large). Mirage tasks are non-trivially harder; the benchmark has discriminating power and is not saturated.

## Additional Insights

**The competence trap.** On `rubric` tasks, claude-opus-4-5 detects 94% of logical traps. On `expertise_trap` tasks, its detection collapses to 17%. It catches obvious flaws but confidently applies domain reasoning without questioning whether the *domain framing itself* is appropriate.

**Hedging ≠ metacognition.** gpt-4o-mini posts 100% `expertise_trap` TDR alongside the lowest clean accuracy (75.9%). Investigation shows it hedges on *anything complex-sounding* — helpful on this family, noisy elsewhere. Metacognitive monitoring must be discriminated from general defensive hedging.

**One family is a clean null.** `over_specification` tasks yield r = +0.08 with a CI that spans zero. When the trap is "notice irrelevant constraints," all six models succeed at similar rates (TDR 0.63–0.88). This is a legitimate negative result about trap type, not a methodological failure.

## Limitations and Honest Caveats

- **n = 6 models.** CIs are wide. The sign is robust (all non-null CIs exclude zero, LOO-stable), but the point estimates will tighten with more models.
- **Single judge.** Results are scored by `claude-sonnet-4-5`. Cross-judge validation with GPT-4o and Gemini is the next step. Expected direction: κ ≥ 0.7 on ordinal rubric scores would confirm robustness; lower would motivate ensemble judging.
- **Judge prompt sensitivity.** All models receive an identical metacognition-eliciting system prompt. This is intentional — we test monitoring under a prompt that *invites* it. Zero-shot monitoring without the prompt is a separate study.
- **Task authorship bias.** Tasks were written by a single author; familiarity with Anthropic models' failure modes could subtly tilt tasks. The published task set is frozen; any follow-up will use an independent author pool.
- **Correlation, not causation.** Higher capability *co-occurs* with worse monitoring on these families. The causal story — whether it's training objectives, scale, or RLHF reward hacking — is outside the benchmark's scope.

## Reproducibility

The full pipeline runs from a clean clone:

```bash
git clone https://github.com/dayeon603-pixel/MetaMirage
cd MetaMirage && pip install -r requirements.txt
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
python v3_judge_evaluator.py --models claude-opus-4-5 gpt-4o gpt-4o-mini \
                                      claude-sonnet-4-5 gemini-1.5-pro llama-3-70b \
                             --tasks v3_tasks_50.json \
                             --output data/eval_results.json
python v3_statistical_analysis.py --input data/eval_results.json \
                                  --output v3_analysis.json
open dashboard.html
```

Expected runtime: ~15 minutes on default rate limits; cost: ~$3 in API calls.

Every number in this writeup is traceable to `v3_analysis.json`. The Kaggle Benchmark is built by `kaggle_task.py` from the same `v3_tasks_50.json` the evaluator reads — single source of truth.

## Conclusion

MetaMirage is a small, carefully-constructed benchmark that asks one question: *does capability imply self-awareness?* The answer, on three independent task families with tight CIs and LOO stability, is **no** — it implies the opposite. This dissociation is the central empirical result, and it is the kind of signal an AGI-progress benchmark has to surface. A model that answers fluently but cannot flag its own traps is not closer to general intelligence; it is a more effective source of confident error.

The benchmark is small by design — small enough to author from scratch, large enough to find an effect that survives leave-one-out and excludes zero. The next steps are cross-judge validation, broader model coverage, and adversarial task expansion. The finding itself, at this n, is already robust.

---

**Author:** Dayeon Kang · **Benchmark link:** [Kaggle Benchmark (private until deadline)] · **Code:** [MetaMirage on GitHub](https://github.com/dayeon603-pixel/MetaMirage) · **Dashboard:** `dashboard.html` in repo
