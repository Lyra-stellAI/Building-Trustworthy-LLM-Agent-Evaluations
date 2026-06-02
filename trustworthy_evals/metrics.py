"""Statistical helpers used throughout the guide.

Pure-stdlib implementations (no numpy/scipy) so every example and test runs
offline with zero third-party dependencies. These are the numbers you actually
report when you validate a judge or an agent:

* :func:`pearson` -- judge-vs-human correlation (Parts 2 & 6).
* :func:`cohens_kappa` / :func:`agreement_rate` -- inter-rater agreement and
  judge-vs-human agreement on categorical verdicts (Parts 5 & 6).
* :func:`pass_at_k` / :func:`pass_hat_k` -- the two agent reliability rates
  that tell opposite stories as k grows (Part 9).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    if not xs:
        raise ValueError("mean() of empty sequence")
    return sum(xs) / len(xs)


def normalize(score: float, lo: float, hi: float) -> float:
    """Rescale a point on a ``[lo, hi]`` scale to ``[0, 1]``.

    A 1-4 judge score of 3 becomes ``(3 - 1) / (4 - 1) = 0.667``; a 1-5 score
    of 4 becomes ``0.75``. This is the reporting normalization used in Parts 2
    and 7.
    """
    if hi == lo:
        raise ValueError("normalize() requires hi != lo")
    return (score - lo) / (hi - lo)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient between two equal-length sequences.

    Returns 0.0 when either series has zero variance (the correlation is
    undefined; 0.0 is the conventional, safe report). This is the headline
    judge-quality number in the tutorial: the naive judge lands at ~0.567, the
    improved judge at ~0.843, against a two-human ceiling of ~0.563.
    """
    xs = list(xs)
    ys = list(ys)
    if len(xs) != len(ys):
        raise ValueError("pearson() requires equal-length inputs")
    n = len(xs)
    if n == 0:
        raise ValueError("pearson() of empty sequence")
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def agreement_rate(a: Sequence[object], b: Sequence[object]) -> float:
    """Fraction of positions where two raters give the identical label."""
    a = list(a)
    b = list(b)
    if len(a) != len(b):
        raise ValueError("agreement_rate() requires equal-length inputs")
    if not a:
        raise ValueError("agreement_rate() of empty sequence")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """Cohen's kappa: agreement corrected for chance.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement and
    ``p_e`` is the agreement expected if both raters labeled independently at
    their observed marginal rates. 1.0 = perfect, 0.0 = chance-level, negative =
    worse than chance. This is the agreement metric PoLL (Part 5) reports when a
    jury of small models beats a single large judge.
    """
    a = list(a)
    b = list(b)
    if len(a) != len(b):
        raise ValueError("cohens_kappa() requires equal-length inputs")
    n = len(a)
    if n == 0:
        raise ValueError("cohens_kappa() of empty sequence")
    p_o = agreement_rate(a, b)
    count_a = Counter(a)
    count_b = Counter(b)
    labels = set(count_a) | set(count_b)
    p_e = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)
    if p_e == 1.0:
        # Both raters used a single label for everything and agreed -> perfect.
        return 1.0
    return (p_o - p_e) / (1 - p_e)


# ---------------------------------------------------------------------------
# Agent reliability rates (Part 9)
# ---------------------------------------------------------------------------


def pass_at_k(p: float, k: int) -> float:
    """``pass@k``: probability of *at least one* success in k attempts.

    ``1 - (1 - p)**k``. Rises with k -- more shots on goal. Use it when one
    success is enough (generate-and-test coding: any patch that passes wins).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability in [0, 1]")
    if k < 1:
        raise ValueError("k must be >= 1")
    return 1.0 - (1.0 - p) ** k


def pass_hat_k(p: float, k: int) -> float:
    """``pass^k``: probability that *all* k trials succeed.

    ``p**k``. Falls with k -- consistency is a harder bar. Use it for
    customer-facing agents where users expect it to work every time. At a 75%
    per-trial rate, ``pass^3 = 0.75**3 ~= 0.42``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability in [0, 1]")
    if k < 1:
        raise ValueError("k must be >= 1")
    return p ** k


def _comb(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def pass_at_k_estimate(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` estimator from samples (the HumanEval estimator).

    Given ``n`` total trials of which ``c`` succeeded, estimate the probability
    that ``k`` randomly drawn trials contain at least one success:

        ``1 - C(n - c, k) / C(n, k)``

    This is less biased than ``1 - (1 - c/n)**k`` for small n, which is why
    benchmark harnesses sample n >> k and use this form.
    """
    if k < 1 or k > n:
        raise ValueError("require 1 <= k <= n")
    if not 0 <= c <= n:
        raise ValueError("require 0 <= c <= n")
    if c == 0:
        return 0.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def success_rate(outcomes: Iterable[bool]) -> float:
    """Per-trial success rate from a sequence of pass/fail booleans."""
    outcomes = list(outcomes)
    if not outcomes:
        raise ValueError("success_rate() of empty sequence")
    return sum(1 for o in outcomes if o) / len(outcomes)


__all__ = [
    "mean",
    "normalize",
    "pearson",
    "agreement_rate",
    "cohens_kappa",
    "pass_at_k",
    "pass_hat_k",
    "pass_at_k_estimate",
    "success_rate",
]
