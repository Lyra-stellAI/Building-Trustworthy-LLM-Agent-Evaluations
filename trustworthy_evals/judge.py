"""Part 2 - Building the judge, and validating it before you trust it.

Two things live here:

* :func:`extract_score` and :class:`Judge` -- the *real* harness. Robust score
  parsing and a thin wrapper that runs any ``LLMClient`` with the naive or
  improved prompt and returns a normalized score. Use these with a real model.
* :func:`simulate_validation_study` -- an offline reproduction of the cookbook's
  central result: a two-human baseline correlation of ~0.56, a naive 0-10 judge
  barely above it at ~0.57, and an improved anchored-rubric judge at ~0.84.

The lesson the numbers teach: never ship a judge you have not measured against
humans, and the ceiling on judge quality is set by how much your humans agree.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .llm import LLMClient
from .metrics import normalize, pearson
from .prompts import IMPROVED_JUDGE_PROMPT, NAIVE_JUDGE_PROMPT


def extract_score(text: str, marker: str = "Total rating:") -> Optional[float]:
    """Pull a numeric score from a chatty judge completion.

    Anchors on the output marker (everything after the *last* occurrence), then
    grabs the first number. Returns ``None`` if nothing parses -- callers decide
    whether that is a retry or a hard failure. This is the parsing recipe from
    Part 2; pair it with structured output in production so it never fails.
    """
    chunk = text.split(marker)[-1] if marker in text else text
    nums = re.findall(r"\d+(?:\.\d+)?", chunk)
    return float(nums[0]) if nums else None


@dataclass
class Judge:
    """Run a real ``LLMClient`` as a rubric/naive judge and normalize the score.

    >>> from trustworthy_evals.llm import ScriptedLLM
    >>> judge = Judge(ScriptedLLM(["Evaluation: solid.\\nTotal rating: 3"]))
    >>> judge.score(question="q?", answer="a")
    0.6666666666666666
    """

    client: LLMClient
    prompt: str = IMPROVED_JUDGE_PROMPT
    scale: Tuple[int, int] = (1, 4)
    marker: str = "Total rating:"

    def score(self, question: str, answer: str) -> float:
        raw = self.client.complete(self.prompt.format(question=question, answer=answer))
        value = extract_score(raw, self.marker)
        if value is None:
            raise ValueError(f"judge produced no parseable score: {raw!r}")
        return normalize(value, *self.scale)

    @classmethod
    def naive(cls, client: LLMClient) -> "Judge":
        """The tempting 0-10 float judge from the start of Part 2. Don't ship it."""
        return cls(client, prompt=NAIVE_JUDGE_PROMPT, scale=(0, 10))


# ---------------------------------------------------------------------------
# Offline validation study (reproduces the 0.567 -> 0.843 story)
# ---------------------------------------------------------------------------


@dataclass
class ValidationStudy:
    true_quality: List[float]
    human_a: List[int]
    human_b: List[int]
    naive_judge: List[float]
    improved_judge: List[int]

    def inter_rater(self) -> float:
        """Pearson correlation between the two human annotators (the ceiling)."""
        return pearson(self.human_a, self.human_b)

    def human_mean(self) -> List[float]:
        return [(a + b) / 2 for a, b in zip(self.human_a, self.human_b)]

    def agreement_subset(self, max_gap: int = 0) -> List[int]:
        """Indices where the two annotators agree within ``max_gap`` points.

        Keeping only these is one of the two noise-reduction routes from Part 2
        (the other is averaging). Low *raw* human agreement means noisy ground
        truth, but the agreement subset has clean labels -- which is what lets a
        good judge correlate well even when raw inter-rater agreement is modest.
        """
        return [i for i, (a, b) in enumerate(zip(self.human_a, self.human_b)) if abs(a - b) <= max_gap]

    def naive_correlation(self, indices: Optional[Sequence[int]] = None) -> float:
        return self._corr(self.naive_judge, indices)

    def improved_correlation(self, indices: Optional[Sequence[int]] = None) -> float:
        return self._corr(self.improved_judge, indices)

    def _corr(self, judge_scores: Sequence[float], indices: Optional[Sequence[int]]) -> float:
        human = self.human_mean()
        if indices is None:
            indices = range(len(human))
        idx = list(indices)
        return pearson([judge_scores[i] for i in idx], [human[i] for i in idx])


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def simulate_validation_study(n: int = 140, seed: int = 0) -> ValidationStudy:
    """Generate a labeled study reproducing the cookbook's correlation story.

    The key structural assumption -- and the reason cleaning helps -- is that
    human disagreement concentrates on genuinely *ambiguous* items. On clear
    items both annotators converge on the same clean label; on ambiguous ones
    they rate noisily and often differ. So raw inter-rater agreement is modest
    (~0.56), yet the agreement subset carries reliable labels, which is what a
    good judge can correlate with at ~0.84.

    Defaults to n=140 for a stable correlation estimate. The cookbook's point
    that "~30 labeled examples is enough" is about getting a usable *read*; the
    estimate is just noisier at that size (try ``n=30`` and several seeds).
    """
    rng = random.Random(seed)
    true_q: List[float] = []
    human_a: List[int] = []
    human_b: List[int] = []
    naive: List[float] = []
    improved: List[int] = []

    p_ambiguous = 0.60  # fraction of items where humans genuinely disagree
    ambiguous_noise = 1.0  # rating noise on ambiguous items
    clear_disagree = 0.15  # small chance clear items still differ by one point
    naive_noise = 0.33  # the naive judge is badly calibrated
    improved_noise = 0.25  # the improved judge tracks quality closely

    for i in range(n):
        q = rng.uniform(0.0, 1.0)
        true_q.append(q)
        clean_label = int(round(_clip(1 + 3 * q, 1, 4)))
        if rng.random() < p_ambiguous:
            a = int(round(_clip(1 + 3 * q + rng.gauss(0, ambiguous_noise), 1, 4)))
            b = int(round(_clip(1 + 3 * q + rng.gauss(0, ambiguous_noise), 1, 4)))
        else:
            a = clean_label
            b = clean_label if rng.random() > clear_disagree else int(
                round(_clip(clean_label + rng.choice([-1, 1]), 1, 4))
            )
        human_a.append(a)
        human_b.append(b)
        naive.append(_clip(round((q + rng.gauss(0, naive_noise)) * 10, 1), 0, 10))
        improved.append(int(round(_clip(1 + 3 * q + rng.gauss(0, improved_noise), 1, 4))))

    return ValidationStudy(true_q, human_a, human_b, naive, improved)


__all__ = [
    "extract_score",
    "Judge",
    "ValidationStudy",
    "simulate_validation_study",
]
