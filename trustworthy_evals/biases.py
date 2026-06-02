"""Part 4 - Known biases and how to mitigate them.

A checklist to design against, demonstrated end to end. Each demo shows the bias
with the default judge, then shows the mitigation shrinking it:

* position bias   -> run both orders and check consistency
* verbosity bias  -> instruct the judge to ignore length (``mitigate=True``)
* self-enhancement-> judge with a different model family than the generator
* distracted eval -> prefer absolute scoring; instruct to ignore tone (Part 3)

The recurring lesson: instruction helps but is not a complete fix, which is why
calibration (Part 6) is non-negotiable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from .llm import Response, SimulatedJudge

# The Part 4 table, as data you can print or assert against.
BIAS_TABLE: List[Tuple[str, str, str]] = [
    ("Position bias", "In pairwise, favors whichever output is first (or last)",
     "Randomize order; run both orders and check consistency"),
    ("Verbosity / length bias", "Longer answers score higher regardless of quality",
     "Explicit rubric instructions on conciseness; length-controlled protocols"),
    ("Self-enhancement bias", "A model rates its own outputs higher",
     "Use a different model family for judging than for generation"),
    ("Provenance / recency bias", "Judge favors stylistic patterns from its training data",
     "Calibrate against human judgments from your domain"),
    ("Distracted evaluation", "Assertive / prolix / sycophantic phrasing sways the verdict",
     "Prefer absolute scoring for verifiable criteria; instruct to ignore tone"),
    ("Intransitivity / choice-set effects", "Cyclic or unstable preferences under comparison",
     "Avoid relative protocols for low-signal data; verify transitivity on a sample"),
]


@dataclass
class BiasDemo:
    name: str
    biased: float  # effect size with the bias active
    mitigated: float  # effect size after the mitigation
    note: str

    @property
    def shrunk(self) -> bool:
        return abs(self.mitigated) < abs(self.biased)


def position_bias_demo(n: int = 400, seed: int = 3) -> BiasDemo:
    """Equal-quality pairs: slot-A win rate (single order) vs consistency (both orders)."""
    judge = SimulatedJudge()
    rng = random.Random(seed)
    slot_a = 0
    consistent = 0
    for i in range(n):
        q = rng.uniform(0.3, 0.8)
        a = Response(text=f"x{i} " + "word " * 30, quality=q)
        b = Response(text=f"y{i} " + "word " * 30, quality=q)
        if judge.compare(a, b) == "A":
            slot_a += 1
        _, ok = judge.compare_both_orders(a, b)
        consistent += int(ok)
    # "biased" = how far single-order win rate is from a fair 0.5.
    # "mitigated" = residual win-rate bias *surfaced* as inconsistency: the
    # both-orders protocol flags the unstable pairs instead of trusting them.
    return BiasDemo(
        name="position",
        biased=slot_a / n - 0.5,
        mitigated=consistent / n - 0.5,
        note=f"single-order slot-A rate={slot_a / n:.3f}; both-orders consistent={consistent / n:.3f} "
        f"(inconsistent pairs are flagged, not trusted)",
    )


def verbosity_bias_demo(n: int = 400, seed: int = 5) -> BiasDemo:
    """Long vs short answers of *equal* quality: mean absolute-score advantage."""
    judge = SimulatedJudge()
    rng = random.Random(seed)
    biased_delta = 0.0
    mitig_delta = 0.0
    for i in range(n):
        q = rng.uniform(0.3, 0.8)
        # Distinct text per item so per-evaluation noise averages out, leaving
        # the length effect (and its removal under mitigation) as the signal.
        short = Response(text=f"s{i} " + "word " * 20, quality=q)
        long = Response(text=f"l{i} " + "word " * 320, quality=q)
        biased_delta += judge.score_absolute(long) - judge.score_absolute(short)
        mitig_delta += judge.score_absolute(long, mitigate=True) - judge.score_absolute(short, mitigate=True)
    return BiasDemo(
        name="verbosity",
        biased=biased_delta / n,
        mitigated=mitig_delta / n,
        note="mean (long - short) absolute score, equal quality; mitigation = 'ignore length' instruction",
    )


def self_enhancement_demo(n: int = 400, seed: int = 11) -> BiasDemo:
    """A judge scores its own family's outputs higher; mitigation = neutral judge."""
    rng = random.Random(seed)
    own_judge = SimulatedJudge(family="claude")  # shares lineage with one generator
    neutral_judge = SimulatedJudge(family="gemini")  # different family than both
    biased_delta = 0.0
    mitig_delta = 0.0
    for i in range(n):
        q = rng.uniform(0.3, 0.8)
        own = Response(text=f"o{i} " + "word " * 30, quality=q, source_model="claude")
        other = Response(text=f"t{i} " + "word " * 30, quality=q, source_model="gpt")
        biased_delta += own_judge.score_absolute(own) - own_judge.score_absolute(other)
        mitig_delta += neutral_judge.score_absolute(own) - neutral_judge.score_absolute(other)
    return BiasDemo(
        name="self_enhancement",
        biased=biased_delta / n,
        mitigated=mitig_delta / n,
        note="mean (own-family - other-family) score, equal quality; mitigation = judge from a third family",
    )


def all_bias_demos() -> List[BiasDemo]:
    return [position_bias_demo(), verbosity_bias_demo(), self_enhancement_demo()]


__all__ = [
    "BIAS_TABLE",
    "BiasDemo",
    "position_bias_demo",
    "verbosity_bias_demo",
    "self_enhancement_demo",
    "all_bias_demos",
]
