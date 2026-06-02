from trustworthy_evals.calibration import (
    EvalTask,
    SamplingPolicy,
    choose_evaluator,
    run_calibration_loop,
)


def test_agreement_rises_over_rounds():
    rounds = run_calibration_loop()
    agreements = [r.agreement for r in rounds]
    # Monotonic non-decreasing, and a clear net gain from the loop.
    assert all(b >= a - 1e-9 for a, b in zip(agreements, agreements[1:]))
    assert agreements[-1] - agreements[0] > 0.1
    # Corrections needed shrink as the judge learns the blind-spot feature.
    assert rounds[-1].n_corrections <= rounds[0].n_corrections


def test_evaluator_routing_prefers_cheapest_capable_tool():
    assert choose_evaluator(EvalTask(mechanical=True)) == "deterministic_rule"
    assert choose_evaluator(EvalTask(has_reference=True)) == "traditional_metric"
    assert choose_evaluator(EvalTask(nuanced=True)) == "llm_judge"
    # High stakes always routes to a human, even when nuanced.
    assert choose_evaluator(EvalTask(high_stakes=True, nuanced=True)) == "human"


def test_sampling_policy():
    policy = SamplingPolicy(sample_rate=0.5, customer_facing_only=True)
    assert policy.should_score(is_customer_facing=True, roll=0.1)
    assert not policy.should_score(is_customer_facing=True, roll=0.9)
    assert not policy.should_score(is_customer_facing=False, roll=0.1)
