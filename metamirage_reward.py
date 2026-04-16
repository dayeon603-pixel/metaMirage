"""
MetaMirage as an RLHF Training Signal
======================================

This module demonstrates how MetaMirage can be used as a reward function
in an RLHF fine-tuning loop. Instead of relying solely on human annotator
preferences (which bias toward confident, decisive answers — the root cause
of the sign-flip), this reward explicitly rewards calibrated metacognitive
monitoring.

Usage in an RLHF loop:
    from metamirage_reward import MetaMirageReward

    reward_fn = MetaMirageReward(tasks_file="v3_tasks_50.json")

    # In your RLHF training loop:
    for batch in training_data:
        response = model.generate(batch.prompt)
        reward = reward_fn.score(batch.prompt, response, batch.task_meta)
        optimizer.step(reward)

The reward combines:
  - Trap detection (did the model flag the flaw?)         weight: 0.50
  - Confidence calibration (was confidence appropriate?)   weight: 0.30
  - Answer quality (was the final answer correct?)         weight: 0.20

This weighting explicitly prioritizes MONITORING over CORRECTNESS,
counteracting the RLHF confidence-pressure bias identified in the
MetaMirage benchmark.

The reward function is differentiable-friendly (returns a scalar float
in [0, 1]) and can be plugged into any PPO / DPO / REINFORCE loop.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from dataclasses import dataclass


TRAP_SIGNALS = [
    "cannot be determined", "cannot determine", "ambiguous", "ambiguity",
    "unclear", "not enough information", "insufficient information",
    "flag", "note that", "should note", "premise", "assumes incorrectly",
    "false premise", "the question", "however", "but wait",
    "unanswerable", "cannot answer", "i cannot", "impossible to determine",
    "missing information", "no basis", "not stated", "not provided",
]

ABSTAIN_SIGNALS = [
    "cannot answer", "cannot be answered", "unanswerable",
    "insufficient information", "not enough information",
    "no way to determine", "impossible to determine",
    "i cannot", "missing context", "cannot determine",
]

CONFIDENCE_KEYWORDS = {
    "very confident": 0.92, "highly confident": 0.92,
    "certain": 0.92, "definitely": 0.92,
    "confident": 0.78, "fairly confident": 0.78,
    "pretty sure": 0.78,
    "not confident": 0.32, "uncertain": 0.32,
    "unsure": 0.32, "low confidence": 0.32,
    "very uncertain": 0.15, "very unsure": 0.15,
    "medium": 0.58, "moderate": 0.58,
}


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for logging / debugging."""
    trap_detected: bool
    confidence_score: float
    answer_score: float
    trap_reward: float
    calibration_reward: float
    answer_reward: float
    total_reward: float
    scoring_mode: str


class MetaMirageReward:
    """
    Reward function for RLHF that scores metacognitive monitoring.

    Weights:
      trap_weight:  importance of trap detection (default 0.50)
      calib_weight: importance of confidence calibration (default 0.30)
      answer_weight: importance of answer correctness (default 0.20)

    These defaults counteract the standard RLHF confidence bias by
    weighting monitoring 2.5× higher than answer quality.
    """

    def __init__(
        self,
        tasks_file: str = "v3_tasks_50.json",
        trap_weight: float = 0.50,
        calib_weight: float = 0.30,
        answer_weight: float = 0.20,
    ):
        self.tasks = {
            t["task_id"]: t
            for t in json.loads(Path(tasks_file).read_text())
        }
        self.trap_weight = trap_weight
        self.calib_weight = calib_weight
        self.answer_weight = answer_weight

    def score(
        self,
        response: str,
        task_id: str | None = None,
        task_meta: dict | None = None,
    ) -> RewardBreakdown:
        """
        Score a model response for metacognitive monitoring quality.

        Args:
            response: the model's full text response
            task_id: lookup key into self.tasks (or provide task_meta directly)
            task_meta: dict with keys: variant, correct_answer, scoring_mode

        Returns:
            RewardBreakdown with .total_reward in [0, 1]
        """
        if task_meta is None:
            task_meta = self.tasks.get(task_id, {})

        text = response.lower()
        variant = task_meta.get("variant", "clean")
        correct = task_meta.get("correct_answer", "").lower()
        mode = task_meta.get("scoring_mode", "rubric")

        # ── Trap detection ──
        trap_detected = variant in ("mirage", "abstain") and any(
            s in text for s in TRAP_SIGNALS
        )
        abstain_detected = variant == "abstain" and any(
            s in text for s in ABSTAIN_SIGNALS
        )

        # ── Confidence extraction ──
        conf = 0.6
        for phrase, val in CONFIDENCE_KEYWORDS.items():
            if phrase in text:
                conf = val
                break
        pct_match = re.search(r"(\d{1,3})\s*%\s*(?:confident|sure|certain)", text)
        if pct_match:
            conf = int(pct_match.group(1)) / 100.0

        # ── Answer correctness ──
        key_terms = [
            w for w in re.findall(r"\b\w{4,}\b", correct)
            if w not in {"that", "this", "with", "from", "have", "been",
                         "will", "would", "could", "should", "their",
                         "there", "they", "what", "which", "task",
                         "cannot", "model", "answer", "question"}
        ][:8]
        matches = sum(1 for t in key_terms if t in text)
        answer_score = matches / max(len(key_terms), 1)

        # ── Reward computation ──
        if variant == "clean":
            trap_reward = 1.0  # no trap to detect
            calib_reward = conf  # higher confidence = better on clean
            answer_reward = answer_score
        elif mode == "abstain_binary":
            trap_reward = 1.0 if abstain_detected else (0.4 if trap_detected else 0.0)
            calib_reward = 1.0 - conf if not abstain_detected else 0.8
            answer_reward = answer_score
        elif mode == "expertise_inverted":
            trap_reward = 1.0 if trap_detected else 0.0
            calib_reward = 1.0 - conf if not trap_detected else 0.5
            answer_reward = answer_score * 0.5  # downweight raw answer quality
        else:  # rubric
            trap_reward = 1.0 if trap_detected else 0.0
            calib_reward = 1.0 - conf if not trap_detected else 0.7
            answer_reward = min(1.0, answer_score + (0.3 if trap_detected else 0.0))

        total = (
            self.trap_weight * trap_reward
            + self.calib_weight * calib_reward
            + self.answer_weight * answer_reward
        )

        return RewardBreakdown(
            trap_detected=trap_detected,
            confidence_score=conf,
            answer_score=answer_score,
            trap_reward=round(trap_reward, 4),
            calibration_reward=round(calib_reward, 4),
            answer_reward=round(answer_reward, 4),
            total_reward=round(total, 4),
            scoring_mode=mode,
        )

    def batch_score(
        self, responses: list[str], task_ids: list[str]
    ) -> list[RewardBreakdown]:
        """Score a batch of responses. Returns list of RewardBreakdowns."""
        return [self.score(r, task_id=tid) for r, tid in zip(responses, task_ids)]

    def as_scalar_reward(self, response: str, task_id: str) -> float:
        """Convenience: returns just the float reward for RLHF integration."""
        return self.score(response, task_id=task_id).total_reward


# ── Example usage ──
if __name__ == "__main__":
    reward_fn = MetaMirageReward()

    # Simulated model response to a mirage task (expertise trap)
    test_response = (
        "I should note that this question contains a false premise. "
        "The p-value of 0.049 with n=10,000 likely reflects a trivially "
        "small effect size. I'm not confident we can determine approval "
        "from this information alone. Medium confidence."
    )

    # Score it
    result = reward_fn.score(
        response=test_response,
        task_id="16177aeb3a",  # expertise_trap task
    )

    print("=== MetaMirage Reward Scoring ===")
    print(f"  Trap detected:      {result.trap_detected}")
    print(f"  Confidence:         {result.confidence_score}")
    print(f"  Answer score:       {result.answer_score}")
    print(f"  ---")
    print(f"  Trap reward:        {result.trap_reward}")
    print(f"  Calibration reward: {result.calibration_reward}")
    print(f"  Answer reward:      {result.answer_reward}")
    print(f"  ---")
    print(f"  TOTAL REWARD:       {result.total_reward}")
    print()
    print("This reward can be plugged into any PPO/DPO/REINFORCE loop as:")
    print("  reward = metamirage_reward.as_scalar_reward(response, task_id)")
    print()
    print("Default weights prioritize monitoring over correctness:")
    print("  trap_detection:  50%  (vs ~0% in standard RLHF)")
    print("  calibration:     30%  (vs ~0% in standard RLHF)")
    print("  answer_quality:  20%  (vs ~100% in standard RLHF)")
    print()
    print("This directly counteracts the RLHF confidence-pressure bias")
    print("identified in the MetaMirage benchmark (Perez 2022, Casper 2023).")
