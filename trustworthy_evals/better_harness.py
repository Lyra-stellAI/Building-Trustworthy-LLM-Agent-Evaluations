"""Part 13.2 - better-harness: closing the loop with evals.

An autonomous harness-optimization loop in which one Deep Agent improves another
agent's harness, keeping a change only if the evals say it generalized. It
operationalizes this whole guide and maps almost one-to-one onto the Part 9
vocabulary.

The structure is the lesson, and it ports to any stack:

* separate the **harness** (the inner agent's editable surfaces) from the **eval
  harness** (the runner that grades);
* hold out a **strata-matched private split** the optimizer cannot see;
* accept a change only when held-out performance improves (greedy hill-climbing
  with a regression guard);
* keep local trace artifacts so failures are auditable.

This is the Part 3 mechanism-design thesis made literal: the optimizer's
objective (the eval) is engineered so the only way to score is to genuinely
improve the harness. A change that memorizes the visible train cases without
generalizing fails the hidden holdout and is rejected. We demonstrate exactly
that: an overfitting "hack" candidate is considered and discarded each iteration,
while generalizing edits are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Eval cases (strata-matched across the train / holdout / scorecard splits)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    id: str
    stratum: str  # 'tool_use' | 'conversation'
    difficulty: float  # passes if the harness capability on this stratum >= difficulty


@dataclass(frozen=True)
class EvalSplit:
    train: Tuple[EvalCase, ...]
    holdout: Tuple[EvalCase, ...]
    scorecard: Tuple[EvalCase, ...]


def default_splits() -> EvalSplit:
    """Train / holdout / scorecard, each covering the same strata (the validator
    enforces this in the real example)."""
    train = (
        EvalCase("tr_tool_1", "tool_use", 0.55),
        EvalCase("tr_tool_2", "tool_use", 0.80),
        EvalCase("tr_tool_3", "tool_use", 0.95),
        EvalCase("tr_conv_1", "conversation", 0.55),
        EvalCase("tr_conv_2", "conversation", 0.82),
        EvalCase("tr_conv_3", "conversation", 0.97),
    )
    holdout = (
        EvalCase("ho_tool_1", "tool_use", 0.66),
        EvalCase("ho_tool_2", "tool_use", 0.88),
        EvalCase("ho_conv_1", "conversation", 0.66),
        EvalCase("ho_conv_2", "conversation", 0.90),
    )
    scorecard = (
        EvalCase("sc_tool_1", "tool_use", 0.70),
        EvalCase("sc_conv_1", "conversation", 0.70),
    )
    return EvalSplit(train, holdout, scorecard)


# ---------------------------------------------------------------------------
# The harness the optimizer edits (its surfaces are the knobs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessConfig:
    base_model: float = 0.45  # frozen weights
    general_policy: float = 0.45  # a surface that helps EVERY stratum (generalizing)
    stratum_surfaces: Tuple[Tuple[str, float], ...] = ()  # per-stratum quality
    memorized: FrozenSet[str] = frozenset()  # case ids hard-coded (overfitting)

    def _surface(self, stratum: str) -> float:
        return dict(self.stratum_surfaces).get(stratum, 0.0)

    def capability(self, case: EvalCase) -> float:
        cap = self.base_model + 0.5 * self.general_policy + 0.45 * self._surface(case.stratum)
        if case.id in self.memorized:
            cap = max(cap, case.difficulty)  # a hard-coded pass for a specific case
        return cap

    def passing(self, cases: Sequence[EvalCase]) -> List[str]:
        return [c.id for c in cases if self.capability(c) >= c.difficulty]

    def n_pass(self, cases: Sequence[EvalCase]) -> int:
        return len(self.passing(cases))

    # -- edits the outer agent can propose ---------------------------------

    def raise_general(self, step: float = 0.15) -> "HarnessConfig":
        return replace(self, general_policy=min(1.0, self.general_policy + step))

    def raise_stratum(self, stratum: str, step: float = 0.45) -> "HarnessConfig":
        current = dict(self.stratum_surfaces)
        current[stratum] = min(1.0, current.get(stratum, 0.0) + step)
        return replace(self, stratum_surfaces=tuple(sorted(current.items())))

    def memorize(self, case_ids: Sequence[str]) -> "HarnessConfig":
        # The overfitting hack: hard-code the visible failures, but the
        # case-specific patch displaces general logic, hurting the policy surface.
        return replace(
            self,
            memorized=self.memorized | frozenset(case_ids),
            general_policy=max(0.0, self.general_policy - 0.20),
        )


# ---------------------------------------------------------------------------
# The optimization loop
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    iteration: int
    candidate: str
    kind: str  # 'generalizing' | 'overfitting'
    accepted: bool
    reason: str
    train_pass: int
    holdout_pass: int


@dataclass
class OptimizationRun:
    decisions: List[Decision]
    baseline_train: int
    baseline_holdout: int
    baseline_scorecard: int
    final_train: int
    final_holdout: int
    final_scorecard: int
    final_config: HarnessConfig

    @property
    def holdout_gain(self) -> int:
        return self.final_holdout - self.baseline_holdout

    def accepted(self) -> List[Decision]:
        return [d for d in self.decisions if d.accepted]

    def rejected_overfits(self) -> List[Decision]:
        return [d for d in self.decisions if d.kind == "overfitting" and not d.accepted]


def _propose(config: HarnessConfig, split: EvalSplit) -> List[Tuple[str, str, HarnessConfig]]:
    """The outer 'better agent' proposes candidate harness edits.

    It can only see the *train* failures. It offers generalizing edits (raise a
    stratum surface, raise the general policy) and one overfitting hack (memorize
    the visible train failures).
    """
    visible_failures = [c for c in split.train if config.capability(c) < c.difficulty]
    failing_strata = sorted({c.stratum for c in visible_failures})

    candidates: List[Tuple[str, str, HarnessConfig]] = []
    for stratum in failing_strata:
        candidates.append((f"raise[{stratum}]", "generalizing", config.raise_stratum(stratum)))
    candidates.append(("raise[general_policy]", "generalizing", config.raise_general()))
    if visible_failures:
        candidates.append(
            ("memorize[visible train failures]", "overfitting", config.memorize([c.id for c in visible_failures]))
        )
    return candidates


def optimize(config: HarnessConfig = None, split: EvalSplit = None, max_iters: int = 8) -> OptimizationRun:
    """Run the better-harness loop and return the full decision history.

    Acceptance rule: keep a candidate only if the combined train+holdout pass
    count strictly improves **and** the private holdout does not regress. The
    held-out non-regression is the generalization guard -- it is what rejects an
    edit that merely memorizes the visible train cases.
    """
    config = config or HarnessConfig()
    split = split or default_splits()

    base_train = config.n_pass(split.train)
    base_holdout = config.n_pass(split.holdout)
    base_scorecard = config.n_pass(split.scorecard)

    decisions: List[Decision] = []
    for it in range(max_iters):
        if config.n_pass(split.train) == len(split.train) and config.n_pass(split.holdout) == len(split.holdout):
            break  # everything passes

        candidates = _propose(config, split)
        if not candidates:
            break

        cur_train = config.n_pass(split.train)
        cur_holdout = config.n_pass(split.holdout)

        # Evaluate every candidate once on train+holdout.
        evaluated = []
        for name, kind, cand in candidates:
            ct = cand.n_pass(split.train)
            ch = cand.n_pass(split.holdout)
            combined_improves = (ct + ch) > (cur_train + cur_holdout)
            holdout_ok = ch >= cur_holdout  # the generalization guard
            eligible = combined_improves and holdout_ok
            evaluated.append((name, kind, cand, ct, ch, eligible, holdout_ok, combined_improves))

        eligible = [e for e in evaluated if e[5]]
        best = max(eligible, key=lambda e: (e[3] + e[4], e[4])) if eligible else None

        # Log exactly one decision per candidate this iteration.
        for e in evaluated:
            name, kind, cand, ct, ch, is_elig, holdout_ok, combined_improves = e
            if best is not None and e is best:
                decisions.append(Decision(it, name, kind, True, "accepted", ct, ch))
            elif not holdout_ok:
                decisions.append(Decision(it, name, kind, False, "holdout regressed (did not generalize)", ct, ch))
            elif not combined_improves:
                decisions.append(Decision(it, name, kind, False, "no combined improvement", ct, ch))
            else:
                decisions.append(Decision(it, name, kind, False, "not selected (a better edit won)", ct, ch))

        if best is None:
            break  # no improving, generalizing change available
        config = best[2]

    return OptimizationRun(
        decisions=decisions,
        baseline_train=base_train,
        baseline_holdout=base_holdout,
        baseline_scorecard=base_scorecard,
        final_train=config.n_pass(split.train),
        final_holdout=config.n_pass(split.holdout),
        final_scorecard=config.n_pass(split.scorecard),
        final_config=config,
    )


__all__ = [
    "EvalCase",
    "EvalSplit",
    "default_splits",
    "HarnessConfig",
    "Decision",
    "OptimizationRun",
    "optimize",
]
