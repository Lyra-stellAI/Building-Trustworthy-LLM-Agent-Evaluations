"""Part 8 - Frameworks: you don't have to build this from scratch.

Educational reimplementations of the two metric shapes the guide highlights:
RAGAS-style component metrics, and DeepEval-style pytest-for-LLMs with G-Eval
(rubric), DAGMetric (deterministic tree), and ArenaGEval (pairwise).
"""

from _bootstrap import header, section

from trustworthy_evals.frameworks import (
    ArenaGEval,
    DAGMetric,
    DAGNode,
    GEval,
    LLMTestCase,
    RagasSample,
    assert_test,
    ragas_score,
)
from trustworthy_evals.llm import Response, SimulatedJudge


def main() -> None:
    header("Part 8 - RAGAS- and DeepEval-style metrics (educational reimplementations)")

    section("RAGAS: four component metrics + their mean")
    sample = RagasSample(
        question="How much cheaper is a panel than a single large judge?",
        answer_claims=["a panel runs over seven times cheaper than a single large judge"],
        context="the panel agrees with humans better and runs over seven times cheaper than a single large judge",
        generated_questions=["how much cheaper is a panel than a large judge"],
        retrieved_relevant=[True, True, False],
        reference_claims_supported=[True, True],
    )
    for k, v in ragas_score(sample).items():
        print(f"  {k:18s}: {v:.2f}")

    section("DeepEval: pytest-style assert_test with a G-Eval rubric metric")
    good = LLMTestCase(input="q", actual_output="correct, grounded answer", expected_output="correct", quality=0.9)
    bad = LLMTestCase(input="q", actual_output="off-topic ramble", expected_output="correct", quality=0.1)
    print(f"  good case GEval score: {GEval().measure(good).score:.2f}  (passes threshold)")
    try:
        assert_test(bad, [GEval(threshold=0.7)])
    except AssertionError as e:
        print(f"  bad case raises      : {e}")

    section("DeepEval: DAGMetric (deterministic decision tree)")
    root = DAGNode(predicate=lambda c: "## Summary" in c.actual_output)
    root.branches = {True: DAGNode(predicate=None, leaf_score=1.0),
                     False: DAGNode(predicate=None, leaf_score=0.0)}
    dag = DAGMetric(root=root)
    with_heading = LLMTestCase("q", "## Summary\nbody")
    without_heading = LLMTestCase("q", "no heading")
    print(f"  with required heading   : score={dag.measure(with_heading).score}")
    print(f"  missing required heading: score={dag.measure(without_heading).score}")

    section("DeepEval: ArenaGEval (pairwise, blinded + position-randomized)")
    arena = ArenaGEval(judge=SimulatedJudge(noise=0.0))
    print(f"  winner of (better vs worse): {arena.pick_winner(Response('b', quality=0.9), Response('w', quality=0.3))}")

    print("\n  Lesson: reach for RAGAS when tuning a RAG retriever+generator; DeepEval for CI-style")
    print("  testing, custom rubric metrics (G-Eval), and deterministic decision-tree scoring (DAG).")


if __name__ == "__main__":
    main()
