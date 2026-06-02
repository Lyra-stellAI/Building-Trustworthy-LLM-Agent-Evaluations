"""Part 8 - Frameworks: you don't have to build this from scratch.

Small, dependency-free *educational* reimplementations of the two metric shapes
the guide highlights. These are not the real libraries -- install ``ragas`` and
``deepeval`` for production -- but they make the API shape and the underlying
mechanics concrete and runnable.

* RAGAS-style component metrics: faithfulness, answer relevancy, context
  precision, context recall -- mostly decomposition + attribution under the hood.
* DeepEval-style: ``LLMTestCase`` + ``assert_test`` (pytest for LLM outputs),
  ``GEval`` (rubric-based, flexible), ``DAGMetric`` (deterministic decision
  tree), and ``ArenaGEval`` (pairwise with the Part 3-4 bias mitigations built in).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .llm import Response, SimulatedJudge
from .rag_eval import tokenize


# ===========================================================================
# RAGAS-style component metrics (reference-free ones run on live traffic)
# ===========================================================================


def _overlap_fraction(a: str, b: str) -> float:
    """Fraction of a's content tokens that appear in b. A cheap stand-in for the
    LLM-judged claim-support check RAGAS performs."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def faithfulness(answer_claims: Sequence[str], context: str, support_threshold: float = 0.5) -> float:
    """Of all claims in the answer, what fraction are supported by the context?

    Reference-free, and the single best catch for hallucination. RAGAS does this
    by decomposing the answer into claims and verifying each against the context;
    here we approximate "supported" with content-token overlap.
    """
    if not answer_claims:
        return 0.0
    supported = sum(1 for c in answer_claims if _overlap_fraction(c, context) >= support_threshold)
    return supported / len(answer_claims)


def answer_relevancy(generated_questions: Sequence[str], original_question: str) -> float:
    """Does the answer address the question?

    RAGAS generates candidate questions *from the answer* and measures their
    similarity to the real one. Reference-free.
    """
    if not generated_questions:
        return 0.0
    return sum(_overlap_fraction(q, original_question) for q in generated_questions) / len(generated_questions)


def context_precision(retrieved_relevant: Sequence[bool]) -> float:
    """Are the relevant chunks ranked at the top of what the retriever returned?

    Mean of precision@k computed at each rank that holds a relevant chunk -- the
    standard ranking-aware precision. Higher when relevant chunks come first.
    """
    relevant = [bool(x) for x in retrieved_relevant]
    if not any(relevant):
        return 0.0
    hits = 0
    precisions = []
    for i, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(precisions)


def context_recall(reference_claims_supported: Sequence[bool]) -> float:
    """Does the retrieved context cover everything in the reference answer?

    Statement-by-statement attribution; needs ground truth.
    """
    claims = [bool(x) for x in reference_claims_supported]
    if not claims:
        return 0.0
    return sum(claims) / len(claims)


@dataclass
class RagasSample:
    question: str
    answer_claims: List[str]
    context: str
    generated_questions: List[str]
    retrieved_relevant: List[bool]
    reference_claims_supported: List[bool]


def ragas_score(sample: RagasSample) -> Dict[str, float]:
    """The classic ragas score: the four core metrics plus their mean."""
    f = faithfulness(sample.answer_claims, sample.context)
    ar = answer_relevancy(sample.generated_questions, sample.question)
    cp = context_precision(sample.retrieved_relevant)
    cr = context_recall(sample.reference_claims_supported)
    return {
        "faithfulness": f,
        "answer_relevancy": ar,
        "context_precision": cp,
        "context_recall": cr,
        "ragas_score": (f + ar + cp + cr) / 4,
    }


# ===========================================================================
# DeepEval-style: pytest for LLM outputs
# ===========================================================================


@dataclass
class LLMTestCase:
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    # Latent quality lets the offline GEval score without a real model.
    quality: Optional[float] = None


@dataclass
class MetricResult:
    name: str
    score: float  # 0-1
    threshold: float
    passed: bool
    reason: str


class Metric:
    """Interface: a metric measures a test case and returns a 0-1 score + verdict."""

    name = "metric"
    threshold = 0.5

    def measure(self, case: LLMTestCase) -> MetricResult:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class GEval(Metric):
    """Rubric-based custom metric. You give a criterion in plain language.

    Real DeepEval expands the criterion into chain-of-thought evaluation steps,
    then scores via a token-probability-weighted sum. Offline, we score the
    case's latent quality (or actual-vs-expected overlap) through the simulated
    judge -- flexible, but not deterministic, exactly as the guide describes.
    """

    name: str = "GEval"
    criteria: str = "Is the actual output consistent with the expected output?"
    threshold: float = 0.5
    judge: SimulatedJudge = field(default_factory=lambda: SimulatedJudge(noise=0.03))

    def measure(self, case: LLMTestCase) -> MetricResult:
        if case.quality is not None:
            quality = case.quality
        elif case.expected_output is not None:
            quality = _overlap_fraction(case.actual_output, case.expected_output)
        else:
            quality = 0.5
        score_1_5 = self.judge.score_absolute(Response(text=case.actual_output, quality=quality), scale=(1, 5))
        score = (score_1_5 - 1) / 4
        passed = score >= self.threshold
        return MetricResult(self.name, score, self.threshold, passed,
                            f"criterion={self.criteria!r}; judged {score_1_5}/5")


@dataclass
class DAGNode:
    """A node in a deterministic decision tree (binary or non-binary)."""

    predicate: Callable[[LLMTestCase], object]
    branches: Dict[object, "DAGNode"] = field(default_factory=dict)
    leaf_score: Optional[float] = None  # set on leaves

    def evaluate(self, case: LLMTestCase) -> float:
        if self.leaf_score is not None:
            return self.leaf_score
        key = self.predicate(case)
        if key not in self.branches:
            raise KeyError(f"DAG has no branch for predicate result {key!r}")
        return self.branches[key].evaluate(case)


@dataclass
class DAGMetric(Metric):
    """Deterministic decision-tree metric: fully reproducible, rule-shaped scoring.

    Use it when the rubric decomposes into yes/no checks ("wrong if any required
    heading is missing; penalize wrong order"). Trades G-Eval's flexibility for
    reproducibility; you can nest a GEval inside a leaf in the real library.
    """

    root: DAGNode
    name: str = "DAGMetric"
    threshold: float = 0.5

    def measure(self, case: LLMTestCase) -> MetricResult:
        score = self.root.evaluate(case)
        return MetricResult(self.name, score, self.threshold, score >= self.threshold,
                            "deterministic decision-tree score")


@dataclass
class ArenaGEval:
    """Pairwise metric with blinding + position randomization built in.

    Picks the better of two contestants using the both-orders aggregation from
    Parts 3-4, so position bias is controlled and inconsistency is surfaced.
    """

    judge: SimulatedJudge = field(default_factory=SimulatedJudge)

    def pick_winner(self, a: Response, b: Response) -> str:
        verdict, consistent = self.judge.compare_both_orders(a, b)
        return {"A": "A", "B": "B", "C": "tie/uncertain"}[verdict] + ("" if consistent else " (inconsistent)")


def assert_test(case: LLMTestCase, metrics: Sequence[Metric]) -> List[MetricResult]:
    """Pytest-style: run every metric and raise if any fails its threshold."""
    results = [m.measure(case) for m in metrics]
    failed = [r for r in results if not r.passed]
    if failed:
        details = "; ".join(f"{r.name}={r.score:.2f}<{r.threshold}" for r in failed)
        raise AssertionError(f"LLMTestCase failed {len(failed)} metric(s): {details}")
    return results


__all__ = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "RagasSample",
    "ragas_score",
    "LLMTestCase",
    "MetricResult",
    "Metric",
    "GEval",
    "DAGNode",
    "DAGMetric",
    "ArenaGEval",
    "assert_test",
]
