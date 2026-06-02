import pytest

from trustworthy_evals.agent_eval import (
    NumericMatch,
    StateCheck,
    TestsPass,
    ToolCall,
    Transcript,
    auth_bypass_task,
    classify_eval_health,
    grade_task,
    likely_broken_task,
    make_flaky_agent,
    run_trials,
    trajectory_match,
)


def _good_transcript():
    return Transcript(
        final_output="Rejected empty/null passwords; auth_blocked logged.",
        tool_calls=[ToolCall("edit_file"), ToolCall("run_tests")],
        final_state={"security_logs": {"event_type": "auth_blocked"}, "quality": 0.85},
        tests_passed={"test_empty_pw_rejected.py": True, "test_null_pw_rejected.py": True},
    )


def test_full_success_passes():
    result = grade_task(auth_bypass_task(), _good_transcript())
    assert result.passed
    assert result.score > 0.9


def test_partial_credit_beats_total_failure():
    task = auth_bypass_task()
    # Identified+verified but the required tests/state failed.
    partial = Transcript(
        final_output="edited",
        tool_calls=[ToolCall("edit_file"), ToolCall("run_tests")],
        final_state={"security_logs": {}, "quality": 0.6},
        tests_passed={"test_empty_pw_rejected.py": True, "test_null_pw_rejected.py": False},
    )
    nothing = Transcript(final_output="", tool_calls=[], final_state={"quality": 0.2}, tests_passed={})
    partial_result = grade_task(task, partial)
    nothing_result = grade_task(task, nothing)
    assert not partial_result.passed  # required gates failed
    assert not nothing_result.passed
    assert partial_result.score > nothing_result.score  # partial credit


def test_state_check_verifies_outcome_not_claim():
    claim_only = Transcript(final_output="your flight is booked", final_state={})
    grader = StateCheck({"reservation": "created"})
    assert not grader.grade(claim_only).passed
    actually = Transcript(final_output="booked", final_state={"reservation": "created"})
    assert grader.grade(actually).passed


def test_numeric_match_tolerance_prevents_reward_hacking_via_rigidity():
    tr = Transcript(final_output="The answer is 96.12")
    assert not NumericMatch(96.124991, tol=0.0).grade(tr).passed  # rigid grader fails a correct answer
    assert NumericMatch(96.124991, tol=0.01).grade(tr).passed


def test_trajectory_match_modes():
    assert trajectory_match(["a", "b", "c"], ["a", "b", "c"], "strict")
    assert not trajectory_match(["b", "a", "c"], ["a", "b", "c"], "strict")
    assert trajectory_match(["b", "a", "c"], ["a", "b", "c"], "unordered")
    assert trajectory_match(["a", "c"], ["a", "b", "c"], "subset")
    assert trajectory_match(["a", "b", "c", "d"], ["a", "c"], "superset")
    with pytest.raises(ValueError):
        trajectory_match(["a"], ["a"], "bogus")


def test_pass_at_k_and_pass_hat_k_tell_opposite_stories():
    report = run_trials(auth_bypass_task(), make_flaky_agent(0.7), k=10)
    assert report.pass_at_k >= report.observed_rate  # at-least-one >= single try
    assert report.pass_hat_k <= report.observed_rate  # all-succeed <= single try
    assert report.pass_at_k > report.pass_hat_k


def test_capability_vs_regression_health():
    assert "headroom" in classify_eval_health("capability", 0.3)
    assert "saturated" in classify_eval_health("capability", 0.95)
    assert classify_eval_health("regression", 0.99) == "healthy"
    assert "REGRESSION" in classify_eval_health("regression", 0.8)


def test_broken_task_heuristic():
    assert likely_broken_task(0.0)
    assert not likely_broken_task(0.42)


def test_tests_pass_partial_score():
    grader = TestsPass(["t1", "t2", "t3", "t4"])
    tr = Transcript(tests_passed={"t1": True, "t2": True, "t3": False, "t4": False})
    result = grader.grade(tr)
    assert result.score == 0.5
    assert not result.passed
