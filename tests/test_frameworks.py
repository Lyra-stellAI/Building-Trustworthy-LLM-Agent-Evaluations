import pytest

from trustworthy_evals.frameworks import (
    ArenaGEval,
    DAGMetric,
    DAGNode,
    GEval,
    LLMTestCase,
    answer_relevancy,
    assert_test,
    context_precision,
    context_recall,
    faithfulness,
    ragas_score,
)
from trustworthy_evals.llm import Response, SimulatedJudge


def test_faithfulness_catches_unsupported_claims():
    ctx = "the panel runs over seven times cheaper than a single large judge"
    supported = ["the panel is seven times cheaper"]
    hallucinated = ["the panel was invented in 1998 by a committee"]
    assert faithfulness(supported, ctx) == 1.0
    assert faithfulness(hallucinated, ctx) < 0.5


def test_context_precision_rewards_top_ranked_relevance():
    # Relevant chunks first scores higher than relevant chunks last.
    assert context_precision([True, True, False]) > context_precision([False, True, True])


def test_context_recall():
    assert context_recall([True, True, False, True]) == 0.75


def test_answer_relevancy():
    assert answer_relevancy(["how cheap is the panel"], "how cheap is the panel") == 1.0


def test_ragas_score_keys():
    from trustworthy_evals.frameworks import RagasSample

    s = RagasSample(
        question="q",
        answer_claims=["q"],
        context="q context",
        generated_questions=["q"],
        retrieved_relevant=[True],
        reference_claims_supported=[True],
    )
    out = ragas_score(s)
    assert set(out) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_score"}


def test_geval_passes_high_quality_fails_low():
    high = GEval(threshold=0.5).measure(LLMTestCase("q", "a", quality=0.95))
    low = GEval(threshold=0.5).measure(LLMTestCase("q", "a", quality=0.05))
    assert high.passed
    assert not low.passed


def test_assert_test_raises_on_failure():
    case = LLMTestCase("q", "a", quality=0.05)
    with pytest.raises(AssertionError):
        assert_test(case, [GEval(threshold=0.9)])


def test_dag_metric_is_deterministic_decision_tree():
    # Score 1.0 if the required heading is present, else 0.0.
    has_heading = DAGNode(predicate=lambda c: "## Summary" in c.actual_output)
    has_heading.branches = {True: DAGNode(predicate=None, leaf_score=1.0),  # type: ignore[arg-type]
                            False: DAGNode(predicate=None, leaf_score=0.0)}  # type: ignore[arg-type]
    metric = DAGMetric(root=has_heading, threshold=0.5)
    assert metric.measure(LLMTestCase("q", "## Summary\nok")).passed
    assert not metric.measure(LLMTestCase("q", "no heading")).passed


def test_arena_geval_picks_better_response():
    arena = ArenaGEval(judge=SimulatedJudge(noise=0.0))
    better = Response(text="better", quality=0.9)
    worse = Response(text="worse", quality=0.3)
    assert arena.pick_winner(better, worse).startswith("A")
