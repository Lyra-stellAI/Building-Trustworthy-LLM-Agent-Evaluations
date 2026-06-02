"""Part 5 - Panels and juries: ablate across judges, then vote.

A single judge is a single point of failure: judge choice silently determines
your results, and a judge carries self-preference bias toward its own family.
The fix (PoLL; Verga et al., 2024) is a jury of several smaller models from
disjoint families, each scoring independently, pooled by a vote.

This module provides the jury machinery and reproduces PoLL's counterintuitive
finding -- a panel of small diverse judges agrees with humans *better* (higher
Cohen's kappa), shows less intra-model bias, and runs many times cheaper than a
single large judge -- via :func:`panel_vs_single_demo`. It also implements the
:func:`judge_ablation` discipline: a conclusion you trust must survive a change
of judge.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .llm import Response, SimulatedJudge
from .metrics import cohens_kappa


def jury_verdict(judges: Sequence[SimulatedJudge], response: Response, mode: str = "mean") -> float:
    """Pool independent judge scores. Mirrors the tutorial's snippet.

    ``mode='majority'`` for discrete verdicts (use an odd number of judges so a
    majority always exists); ``mode='mean'`` (average pooling) for graded scores,
    because a small panel rarely produces a clean majority on a continuous scale.
    """
    votes = [j.score_absolute(response) for j in judges]
    if mode == "majority":
        return Counter(votes).most_common(1)[0][0]
    if mode == "mean":
        return sum(votes) / len(votes)
    raise ValueError("mode must be 'majority' or 'mean'")


@dataclass
class Jury:
    judges: List[SimulatedJudge]
    mode: str = "mean"
    cost_per_judge: float = 0.045  # relative to a single GPT-4-class judge at 1.0

    def score(self, response: Response) -> float:
        return jury_verdict(self.judges, response, self.mode)

    @property
    def cost(self) -> float:
        return self.cost_per_judge * len(self.judges)

    def disagreement(self, response: Response) -> float:
        """Spread of the panel's votes -- treat high disagreement as a route-to-human signal."""
        votes = [j.score_absolute(response) for j in self.judges]
        return max(votes) - min(votes)


def _bucket(score: float, edges: Sequence[float]) -> int:
    """Map a continuous score to an ordinal bucket for kappa (categorical agreement)."""
    for i, e in enumerate(edges):
        if score <= e:
            return i
    return len(edges)


@dataclass
class PanelComparison:
    single_kappa: float
    panel_kappa: float
    single_cost: float
    panel_cost: float

    @property
    def cost_ratio(self) -> float:
        return self.single_cost / self.panel_cost

    @property
    def panel_wins(self) -> bool:
        return self.panel_kappa > self.single_kappa


def panel_vs_single_demo(n: int = 600, seed: int = 0) -> PanelComparison:
    """Reproduce PoLL: a diverse panel beats one big judge on human agreement.

    Setup matches the danger the tutorial calls out: you are evaluating *your
    own system's* outputs (family "claude"), and your single judge happens to
    share that lineage. Self-preference then inflates the *entire* eval set --
    correlated error you cannot prompt away -- so its bucketed scores drift off
    the human label. A panel of three disjoint families has only one member
    sharing lineage, so the inflation is diluted ~3x and the panel tracks humans
    far more faithfully, at a fraction of the cost.
    """
    rng = random.Random(seed)
    system_family = "claude"  # the system under test

    # Single large judge: low noise, but it shares lineage with the system.
    single = SimulatedJudge(family=system_family, noise=0.05, self_pref_w=0.15, length_w=0.0, position_w=0.0)

    # Panel of three smaller (noisier) judges from disjoint families. Only the
    # "claude" member shares lineage, so the self-preference dilutes ~3x.
    panel = Jury(
        judges=[
            SimulatedJudge(family="claude", noise=0.10, self_pref_w=0.15, length_w=0.0, position_w=0.0, seed=1),
            SimulatedJudge(family="gpt", noise=0.10, self_pref_w=0.15, length_w=0.0, position_w=0.0, seed=2),
            SimulatedJudge(family="gemini", noise=0.10, self_pref_w=0.15, length_w=0.0, position_w=0.0, seed=3),
        ],
        mode="mean",
    )

    edges = [1.5, 2.5, 3.5, 4.5]  # 1-5 score -> 5 ordinal buckets
    human_labels: List[int] = []
    single_labels: List[int] = []
    panel_labels: List[int] = []

    for i in range(n):
        q = rng.uniform(0.0, 1.0)
        r = Response(text=f"r{i} " + "word " * 30, quality=q, source_model=system_family)
        # Human label: a clean bucketing of true quality on the 1-5 scale.
        human_labels.append(_bucket(1 + 4 * q, edges))
        single_labels.append(_bucket(single.score_absolute(r, scale=(1, 5)), edges))
        panel_labels.append(_bucket(panel.score(r), edges))

    return PanelComparison(
        single_kappa=cohens_kappa(single_labels, human_labels),
        panel_kappa=cohens_kappa(panel_labels, human_labels),
        single_cost=1.0,
        panel_cost=panel.cost,
    )


def judge_ablation(
    judges: Sequence[SimulatedJudge],
    conclusion: Callable[[SimulatedJudge], bool],
) -> Tuple[bool, Dict[str, bool]]:
    """Run a conclusion under every judge; it is robust only if all agree.

    Example: ``conclusion = lambda j: mean_score(j, variant_b) > mean_score(j, variant_a)``.
    If "B beats A" holds under GPT, Claude, and Qwen you have a real result; if
    it flips when you swap the judge, you measured one model's taste.
    """
    per_judge = {f"{j.family}#{i}": bool(conclusion(j)) for i, j in enumerate(judges)}
    robust = len(set(per_judge.values())) == 1
    return robust, per_judge


__all__ = [
    "jury_verdict",
    "Jury",
    "PanelComparison",
    "panel_vs_single_demo",
    "judge_ablation",
]
