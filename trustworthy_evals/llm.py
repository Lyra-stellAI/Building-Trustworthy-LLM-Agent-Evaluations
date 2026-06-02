"""The judge layer: a deterministic simulator plus real-LLM adapters.

The guide makes claims you can *measure* -- "pairwise flips ~35% of verdicts
under a distractor, absolute only ~9%", "a jury of small models beats one big
judge", "longer answers score higher regardless of quality". To let you see
those effects without API keys or network access, this module ships a
:class:`SimulatedJudge`: a transparent, seeded model of judge behavior whose
biases are explicit knobs.

Two honesty notes:

1.  The simulator does **not** read the prompt. It works off latent attributes
    of a :class:`Response` (its true ``quality`` plus style features). Its job
    is to reproduce *documented* judge behavior so the harness code around it --
    parsing, both-orders aggregation, jury voting, calibration, metrics -- can
    be exercised for real.
2.  We model the COLM 2025 finding **directly**: the comparison protocol is more
    distractor-sensitive than absolute scoring (separate ``*_pairwise`` vs
    ``*_absolute`` weights). The simulation's value is letting you watch the
    *consequences* (flip rates, tie rejection, leaderboard hacking) and run the
    real mitigations against them.

For a real evaluation, swap in :class:`AnthropicJudge` or :class:`OpenAIJudge`
(or any object satisfying :class:`LLMClient`). They format the *same* prompt
templates from :mod:`trustworthy_evals.prompts` and parse scores with the same
:func:`~trustworthy_evals.judge.extract_score` used everywhere else.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

# Pairwise verdict tokens (mirrors the [[A]] / [[B]] / [[C]] convention).
A_WINS = "A"
B_WINS = "B"
TIE = "C"

# The three content-neutral distractors from the COLM 2025 paper (Part 3).
DISTRACTORS = ("assertiveness", "prolixity", "sycophancy")


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class Response:
    """A candidate answer with latent ground-truth attributes.

    In a real pipeline you only observe ``text``; ``quality`` and the style
    features are what a perfect oracle would know. The simulator uses them so we
    can construct controlled experiments (e.g. "same quality, more assertive").
    """

    text: str
    quality: float = 0.5  # latent true quality in [0, 1]
    assertiveness: float = 0.0  # confident, authoritative phrasing  [0, 1]
    prolixity: float = 0.0  # dense, "expert-sounding" verbosity     [0, 1]
    sycophancy: float = 0.0  # overly agreeable, flattering tone      [0, 1]
    length: int = 0  # token-ish length, drives Part-4 verbosity bias
    source_model: str = ""  # generating model family (self-preference bias)

    def __post_init__(self) -> None:
        if self.length == 0 and self.text:
            self.length = len(self.text.split())

    def with_distractor(self, kind: str, strength: float = 1.0) -> "Response":
        """Return a copy with one distractor boosted but ``quality`` unchanged.

        This is the paper's construction ``y -> y_f`` such that ``p(y) =
        p(y_f)``: the modifier changes only style, never facts. A perfectly
        calibrated judge would return the same verdict; a real one often does
        not.
        """
        if kind not in DISTRACTORS:
            raise ValueError(f"unknown distractor {kind!r}; choose from {DISTRACTORS}")
        bumped = {kind: min(1.0, getattr(self, kind) + strength)}
        # Prolixity also reads as extra length, feeding Part-4 verbosity bias.
        if kind == "prolixity":
            bumped["length"] = self.length + int(60 * strength)
        return replace(self, **bumped)


# ---------------------------------------------------------------------------
# Client protocol + a scripted stand-in for tests
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Anything that turns a prompt into text. Real adapters satisfy this."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        ...


class ScriptedLLM:
    """Returns canned completions in order; raises when exhausted.

    Useful for unit-testing the *parsing* path (Part 2) against exactly the kind
    of chatty, marker-wrapped output a real model emits.
    """

    def __init__(self, scripted: Sequence[str]):
        self._queue = list(scripted)
        self.calls: List[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._queue:
            raise RuntimeError("ScriptedLLM ran out of scripted responses")
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# The simulated judge
# ---------------------------------------------------------------------------


def _stable_unit(*parts: object) -> float:
    """A deterministic pseudo-random float in [0, 1) from the given keys.

    Python's built-in ``hash`` is salted per process; we need reproducibility
    across runs, so we hash a stable string representation instead.
    """
    h = hashlib.sha256("\x1f".join(map(repr, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


@dataclass
class SimulatedJudge:
    """A judge whose biases are explicit, documented knobs.

    Every weight defaults near the literature's central estimates so the bundled
    examples reproduce the guide's numbers out of the box. Crank a weight to 0
    to "mitigate" that bias and watch the effect disappear -- which is exactly
    the experiment Parts 3-5 ask you to run.
    """

    family: str = "generic"

    # Distractor sensitivity, modeled per-protocol (the COLM 2025 asymmetry).
    assertiveness_pairwise: float = 0.29
    prolixity_pairwise: float = 0.25
    sycophancy_pairwise: float = 0.22
    assertiveness_absolute: float = 0.045
    prolixity_absolute: float = 0.040
    sycophancy_absolute: float = 0.035

    # Other documented biases (Part 4).
    length_w: float = 0.12  # verbosity/length bias
    position_w: float = 0.05  # favors whichever response is shown first
    self_pref_w: float = 0.12  # self-enhancement: rates its own family higher

    noise: float = 0.02  # judge stochasticity (std-dev, normalized units)
    # Pairwise's tie band is deliberately tiny: comparative judges almost never
    # call a tie even when outputs are equal -- they manufacture a winner. See
    # the equal-quality demo in Part 3.
    pairwise_tie_band: float = 0.002  # |margin| below this -> a tie verdict
    seed: int = 0

    # -- internal helpers ---------------------------------------------------

    def _noise(self, *keys: object) -> float:
        if self.noise <= 0:
            return 0.0
        # Box-Muller from two stable uniforms -> deterministic gaussian.
        u1 = max(1e-9, _stable_unit(self.seed, "n1", *keys))
        u2 = _stable_unit(self.seed, "n2", *keys)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return self.noise * z

    def _length_norm(self, length: int) -> float:
        # Saturating: more words help, with diminishing returns.
        return min(1.0, length / 400.0)

    def _self_pref(self, response: Response) -> float:
        if response.source_model and response.source_model == self.family:
            return self.self_pref_w
        return 0.0

    def _perceived_absolute(self, response: Response, *, mitigate: bool = False) -> float:
        d = 0.0 if mitigate else 1.0  # mitigation = "ignore tone/length" instruction
        return (
            response.quality
            + d * self.assertiveness_absolute * response.assertiveness
            + d * self.prolixity_absolute * response.prolixity
            + d * self.sycophancy_absolute * response.sycophancy
            + d * self.length_w * self._length_norm(response.length)
            + self._self_pref(response)
            + self._noise("abs", response.text)
        )

    def _perceived_pairwise(self, response: Response, slot: str, *, mitigate: bool = False) -> float:
        d = 0.0 if mitigate else 1.0
        return (
            response.quality
            + d * self.assertiveness_pairwise * response.assertiveness
            + d * self.prolixity_pairwise * response.prolixity
            + d * self.sycophancy_pairwise * response.sycophancy
            + d * self.length_w * self._length_norm(response.length)
            + self._self_pref(response)
            + self._noise("pw", slot, response.text)
        )

    # -- public API ---------------------------------------------------------

    def score_absolute(
        self, response: Response, scale: Tuple[int, int] = (1, 5), *, mitigate: bool = False
    ) -> int:
        """Rubric-based (pointwise) score on an anchored integer scale.

        The integer rounding is load-bearing: small stylistic perturbations
        rarely cross a rounding boundary, which is *why* absolute scoring keeps
        penalizing real errors even when the prose is slick (Part 3).
        """
        lo, hi = scale
        perceived = self._perceived_absolute(response, mitigate=mitigate)
        raw = lo + perceived * (hi - lo)
        return int(max(lo, min(hi, round(raw))))

    def compare(
        self,
        a: Response,
        b: Response,
        *,
        mitigate: bool = False,
        apply_position_bias: bool = True,
    ) -> str:
        """Pairwise verdict for (A first, B second). Returns ``A``/``B``/``C``.

        The position term rewards whatever sits in slot A -- the signature
        failure mode of comparative judging. Run :meth:`compare_both_orders` to
        cancel it.
        """
        pa = self._perceived_pairwise(a, "A", mitigate=mitigate)
        pb = self._perceived_pairwise(b, "B", mitigate=mitigate)
        margin = pa - pb
        if apply_position_bias:
            margin += self.position_w  # slot A bonus
        if abs(margin) < self.pairwise_tie_band:
            return TIE
        return A_WINS if margin > 0 else B_WINS

    def compare_both_orders(
        self, a: Response, b: Response, *, mitigate: bool = False
    ) -> Tuple[str, bool]:
        """Run the comparison in both orders and aggregate.

        Returns ``(verdict, consistent)`` where ``verdict`` is in terms of the
        original A/B and ``consistent`` is whether the two orders agreed. This
        is the operational defense the paper itself uses: it cancels position
        bias *and* surfaces instability as a flagged inconsistency.
        """
        v1 = self.compare(a, b, mitigate=mitigate)  # (A, B)
        v2 = self.compare(b, a, mitigate=mitigate)  # (B, A) -> flip to A/B frame
        v2_in_ab = {A_WINS: B_WINS, B_WINS: A_WINS, TIE: TIE}[v2]
        if v1 == v2_in_ab:
            return v1, True
        # Disagreement: the order decided it. Report a tie and flag it.
        return TIE, False


# ---------------------------------------------------------------------------
# Real-LLM adapters (optional; require the relevant SDK + an API key)
# ---------------------------------------------------------------------------


class AnthropicJudge:
    """Thin adapter so the real path uses the same prompts and parser.

    Requires ``pip install anthropic`` and ``ANTHROPIC_API_KEY``. Not exercised
    in the offline test suite -- it documents the production wiring.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", client: Optional[object] = None):
        self.model = model
        self._client = client

    def _ensure(self):  # pragma: no cover - requires network/SDK
        if self._client is None:
            import anthropic  # noqa: WPS433

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, prompt: str) -> str:  # pragma: no cover - requires network/SDK
        client = self._ensure()
        msg = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


class OpenAIJudge:
    """Thin adapter for OpenAI-family judges. Requires ``pip install openai``."""

    def __init__(self, model: str = "gpt-4o-mini", client: Optional[object] = None):
        self.model = model
        self._client = client

    def _ensure(self):  # pragma: no cover - requires network/SDK
        if self._client is None:
            import openai  # noqa: WPS433

            self._client = openai.OpenAI()
        return self._client

    def complete(self, prompt: str) -> str:  # pragma: no cover - requires network/SDK
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


__all__ = [
    "A_WINS",
    "B_WINS",
    "TIE",
    "DISTRACTORS",
    "Response",
    "LLMClient",
    "ScriptedLLM",
    "SimulatedJudge",
    "AnthropicJudge",
    "OpenAIJudge",
]
