"""Part 3 - Pairwise vs. rubric-based scoring, and which to trust.

The conventional wisdom (LangChain) is that pairwise comparison is more reliable
than absolute scoring. The COLM 2025 paper (Tripathi et al.) shows that for
verifiable / correctness-oriented tasks it is backwards: pairwise is the more
vulnerable protocol.

This module reproduces the paper's experiments against the simulated judge:

* :func:`run_distracted_experiment` -- inject a content-neutral distractor into
  the dis-preferred response and measure how often each protocol's verdict
  flips (~35% pairwise vs ~9% absolute in the paper).
* :func:`tie_recognition_experiment` -- on equal-quality pairs, absolute scoring
  recognizes the tie; pairwise manufactures a winner.
* :func:`leaderboard_hacking_demo` -- rewrite low-ranked models to be more
  assertive (no facts changed) and watch them climb a pairwise leaderboard
  while an absolute leaderboard barely moves.

The practical rule the module is built to demonstrate: default to *absolute*
scoring for verifiable criteria and near-tie-heavy data; reserve *pairwise* for
open-ended quality where you lack an anchor, and then run both orders.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .llm import A_WINS, DISTRACTORS, Response, SimulatedJudge

Pair = Tuple[Response, Response]  # (preferred, dis-preferred)


def make_paired_dataset(n: int = 800, seed: int = 42, gap: Tuple[float, float] = (0.05, 0.75)) -> List[Pair]:
    """Build (preferred, dis-preferred) pairs with a spread of quality gaps.

    The gap distribution deliberately includes many small gaps -- the
    low-preference-strength regime where pairwise is most fragile.
    """
    rng = random.Random(seed)
    pairs: List[Pair] = []
    for i in range(n):
        q_pref = rng.uniform(0.20, 0.95)
        q_disp = max(0.02, q_pref - rng.uniform(*gap))
        filler = "word " * rng.randint(20, 60)
        pairs.append(
            (
                Response(text=f"pref{i} {filler}", quality=q_pref),
                Response(text=f"disp{i} {filler}", quality=q_disp),
            )
        )
    return pairs


@dataclass
class DistractedResult:
    distractor: str
    pairwise_flip_rate: float
    absolute_flip_rate: float
    n_pairwise: int
    n_absolute: int


def run_distracted_experiment(
    judge: SimulatedJudge, pairs: Sequence[Pair], distractor: str
) -> DistractedResult:
    """Measure verdict-flip rates when a distractor is added to the worse answer.

    For each pair where the protocol *correctly* prefers the better answer, we
    add the distractor to the dis-preferred answer (quality unchanged) and check
    whether the protocol stops preferring the better answer. Position bias is
    switched off so we isolate the distractor's effect.
    """
    if distractor not in DISTRACTORS:
        raise ValueError(f"unknown distractor {distractor!r}; choose from {DISTRACTORS}")

    pw_flips = pw_base = 0
    ab_flips = ab_base = 0
    for pref, disp in pairs:
        # Pairwise
        if judge.compare(pref, disp, apply_position_bias=False) == A_WINS:
            pw_base += 1
            disp_d = disp.with_distractor(distractor)
            if judge.compare(pref, disp_d, apply_position_bias=False) != A_WINS:
                pw_flips += 1
        # Absolute
        s_pref = judge.score_absolute(pref)
        s_disp = judge.score_absolute(disp)
        if s_pref > s_disp:
            ab_base += 1
            disp_d = disp.with_distractor(distractor)
            if judge.score_absolute(disp_d) >= s_pref:
                ab_flips += 1

    return DistractedResult(
        distractor=distractor,
        pairwise_flip_rate=pw_flips / pw_base if pw_base else 0.0,
        absolute_flip_rate=ab_flips / ab_base if ab_base else 0.0,
        n_pairwise=pw_base,
        n_absolute=ab_base,
    )


def summarize_distracted(
    judge: Optional[SimulatedJudge] = None, pairs: Optional[Sequence[Pair]] = None
) -> Dict[str, DistractedResult]:
    """Run the distracted-evaluation experiment for all three distractors."""
    judge = judge or SimulatedJudge()
    pairs = pairs if pairs is not None else make_paired_dataset()
    return {d: run_distracted_experiment(judge, pairs, d) for d in DISTRACTORS}


def tie_recognition_experiment(
    judge: Optional[SimulatedJudge] = None, n: int = 500, seed: int = 7
) -> Tuple[float, float]:
    """On equal-quality pairs, how often does each protocol recognize the tie?

    Returns ``(absolute_identical_rate, pairwise_tie_rate)``. The paper finds
    absolute assigns identical scores 84-93% of the time while pairwise calls a
    tie only 2-7% of the time -- it manufactures a winner where none exists.
    """
    judge = judge or SimulatedJudge()
    rng = random.Random(seed)
    same_score = pw_tie = 0
    for i in range(n):
        q = rng.uniform(0.2, 0.9)
        filler = "word " * rng.randint(20, 60)
        a = Response(text=f"a{i} {filler}", quality=q)
        b = Response(text=f"b{i} {filler}", quality=q)
        if judge.score_absolute(a) == judge.score_absolute(b):
            same_score += 1
        if judge.compare(a, b, apply_position_bias=False) == "C":
            pw_tie += 1
    return same_score / n, pw_tie / n


# ---------------------------------------------------------------------------
# Leaderboard hacking (the mechanism-design argument)
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    name: str
    quality: float  # latent true quality
    assertive: bool = False  # has the "make it assertive" hack been applied?

    def respond(self, prompt_idx: int, rng: random.Random) -> Response:
        q = max(0.02, min(0.99, self.quality + rng.gauss(0, 0.03)))
        r = Response(text=f"{self.name}/p{prompt_idx} " + "word " * 30, quality=q, source_model=self.name)
        return r.with_distractor("assertiveness") if self.assertive else r


def _pairwise_winrate(models: Sequence[ModelEntry], n_prompts: int, judge: SimulatedJudge, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    wins = {m.name: 0.0 for m in models}
    games = {m.name: 0 for m in models}
    for p in range(n_prompts):
        responses = {m.name: m.respond(p, rng) for m in models}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                mi, mj = models[i], models[j]
                verdict, _ = judge.compare_both_orders(responses[mi.name], responses[mj.name])
                games[mi.name] += 1
                games[mj.name] += 1
                if verdict == A_WINS:
                    wins[mi.name] += 1
                elif verdict == "B":
                    wins[mj.name] += 1
                else:  # tie
                    wins[mi.name] += 0.5
                    wins[mj.name] += 0.5
    return {name: wins[name] / games[name] for name in wins}


def _absolute_mean(models: Sequence[ModelEntry], n_prompts: int, judge: SimulatedJudge, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    totals = {m.name: 0.0 for m in models}
    for p in range(n_prompts):
        for m in models:
            totals[m.name] += judge.score_absolute(m.respond(p, rng))
    return {name: totals[name] / n_prompts for name in totals}


def _ranking(scores: Dict[str, float]) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


@dataclass
class LeaderboardHack:
    pairwise_before: List[str]
    pairwise_after: List[str]
    absolute_before: List[str]
    absolute_after: List[str]
    hacked: List[str]

    def pairwise_rank_change(self, name: str) -> int:
        """Positions gained on the pairwise board after the hack (positive = up)."""
        return self.pairwise_before.index(name) - self.pairwise_after.index(name)

    def absolute_rank_change(self, name: str) -> int:
        return self.absolute_before.index(name) - self.absolute_after.index(name)


def leaderboard_hacking_demo(
    judge: Optional[SimulatedJudge] = None, n_prompts: int = 60, seed: int = 0, hack_bottom: int = 3
) -> LeaderboardHack:
    """Rewrite the lowest-ranked models to be assertive and re-rank.

    No facts change -- only assertiveness. The pairwise leaderboard reshuffles
    (real ranking gains from a content-free exploit); the absolute leaderboard
    barely moves. This is the paper's demonstration that any leaderboard built
    on pairwise preferences can be gamed.
    """
    judge = judge or SimulatedJudge()
    models = [
        ModelEntry("alpha", 0.78),
        ModelEntry("bravo", 0.70),
        ModelEntry("charlie", 0.63),
        ModelEntry("delta", 0.57),
        ModelEntry("echo", 0.52),
        ModelEntry("foxtrot", 0.47),
    ]

    pw_before = _pairwise_winrate(models, n_prompts, judge, seed)
    ab_before = _absolute_mean(models, n_prompts, judge, seed)
    ranking_before = _ranking(pw_before)

    # Hack: take the current bottom-k pairwise models and make them assertive.
    hacked = ranking_before[-hack_bottom:]
    hacked_models = [
        ModelEntry(m.name, m.quality, assertive=(m.name in hacked)) for m in models
    ]

    pw_after = _pairwise_winrate(hacked_models, n_prompts, judge, seed)
    ab_after = _absolute_mean(hacked_models, n_prompts, judge, seed)

    return LeaderboardHack(
        pairwise_before=_ranking(pw_before),
        pairwise_after=_ranking(pw_after),
        absolute_before=_ranking(ab_before),
        absolute_after=_ranking(ab_after),
        hacked=hacked,
    )


__all__ = [
    "Pair",
    "make_paired_dataset",
    "DistractedResult",
    "run_distracted_experiment",
    "summarize_distracted",
    "tie_recognition_experiment",
    "ModelEntry",
    "LeaderboardHack",
    "leaderboard_hacking_demo",
]
