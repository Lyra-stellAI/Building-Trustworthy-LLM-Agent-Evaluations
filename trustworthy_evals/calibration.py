"""Part 6 - Calibration: turning a judge into a trustworthy instrument.

Prompt iteration alone won't close the gap between a technically correct
evaluator and one you trust for shipping decisions. The fix is a systematic
loop, not intuition:

1. Collect human corrections on a sample of traces.
2. Build few-shot examples from those corrections.
3. Track agreement over time and iterate against data.

:func:`run_calibration_loop` simulates this: a judge starts with a blind spot
(it ignores a quality-relevant feature humans care about, e.g. a missing policy
detail), and each round of corrections teaches it more of that feature, so
judge-human agreement climbs round over round.

:func:`choose_evaluator` encodes the broader stack discipline -- deterministic
rules first, traditional metrics when you have a reference, an LLM judge only
for nuance, humans to calibrate -- and :class:`SamplingPolicy` covers the
offline/online operational controls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

# The data flywheel, as the guide frames it.
DATA_FLYWHEEL = (
    "production traces -> usage insights -> datasets -> evals -> improvements -> better traces"
)


@dataclass
class CalibrationRound:
    round_index: int
    n_corrections: int
    feature_coverage: float  # how much of the blind-spot feature the judge now applies
    agreement: float  # judge-vs-human agreement after this round


def run_calibration_loop(
    rounds: int = 5,
    n_items: int = 400,
    p_blind_spot: float = 0.65,
    penalty: float = 0.45,
    learn_rate: float = 0.55,
    seed: int = 0,
) -> List[CalibrationRound]:
    """Simulate the collect -> few-shot -> track loop and return per-round agreement.

    The judge initially scores on surface quality ``q`` and ignores a penalty
    that blind-spot items genuinely deserve (humans apply it). Each round, human
    corrections are turned into few-shot examples, so the judge applies a growing
    fraction of the penalty (``feature_coverage``). Agreement with humans rises
    accordingly -- the concrete payoff of the loop.
    """
    rng = random.Random(seed)
    # Fixed evaluation slice we re-score each round to track agreement honestly.
    items = []
    for _ in range(n_items):
        q = rng.uniform(0.0, 1.0)
        blind = rng.random() < p_blind_spot
        items.append((q, blind))

    def human_label(q: float, blind: bool) -> bool:
        # Humans see the true quality, penalty included.
        effective = q - (penalty if blind else 0.0)
        return effective >= 0.5

    def judge_label(q: float, blind: bool, coverage: float) -> bool:
        # The judge applies only the fraction of the penalty it has learned.
        perceived = q - (penalty * coverage if blind else 0.0)
        return perceived >= 0.5

    results: List[CalibrationRound] = []
    for r in range(rounds):
        coverage = 1.0 - (1.0 - learn_rate) ** r  # 0 at round 0, ->1 as r grows
        agree = sum(
            1 for q, blind in items if judge_label(q, blind, coverage) == human_label(q, blind)
        ) / n_items
        # Corrections collected this round target the still-misjudged blind-spot items.
        n_corr = sum(
            1
            for q, blind in items
            if blind and judge_label(q, blind, coverage) != human_label(q, blind)
        )
        results.append(CalibrationRound(r, n_corr, coverage, agree))
    return results


# ---------------------------------------------------------------------------
# Where an LLM judge fits in the stack (Part 6)
# ---------------------------------------------------------------------------


@dataclass
class EvalTask:
    mechanical: bool = False  # format/length/required-field checks
    has_reference: bool = False  # a gold answer / reference exists
    nuanced: bool = False  # helpfulness, tone, multi-turn on-topic-ness
    high_stakes: bool = False  # medical/legal/product; or building a golden set


def choose_evaluator(task: EvalTask) -> str:
    """Pick the cheapest evaluator that can actually do the job.

    The order matters: deterministic rules are fast, free, and perfectly
    reliable for what they check, so they go first; reach for an LLM judge only
    for the nuanced dimensions rules and reference metrics can't reach; reserve
    humans for high-stakes calls and for calibrating the judges themselves.
    """
    if task.high_stakes:
        return "human"
    if task.mechanical:
        return "deterministic_rule"
    if task.has_reference and not task.nuanced:
        return "traditional_metric"  # exact match / semantic similarity
    if task.nuanced:
        return "llm_judge"
    return "traditional_metric"


# ---------------------------------------------------------------------------
# Offline vs. online operational controls (Part 6)
# ---------------------------------------------------------------------------


@dataclass
class SamplingPolicy:
    """Online-evaluation controls: sample a fraction, target customer-facing traffic.

    Turning on an online evaluator can bump traces into extended retention with
    cost implications, so you sample and target rather than scoring everything.
    """

    sample_rate: float = 1.0
    customer_facing_only: bool = False

    def should_score(self, *, is_customer_facing: bool, roll: float) -> bool:
        if self.customer_facing_only and not is_customer_facing:
            return False
        return roll < self.sample_rate


__all__ = [
    "DATA_FLYWHEEL",
    "CalibrationRound",
    "run_calibration_loop",
    "EvalTask",
    "choose_evaluator",
    "SamplingPolicy",
]
