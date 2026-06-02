"""Part 9 - Evaluating agents, not just outputs.

Agents are systems: they plan, call tools, keep state, and only eventually
return something. Grade the OUTCOME over the trajectory, mix grader families,
build in partial credit, and report a RATE (pass@k / pass^k), not a verdict.
"""

from _bootstrap import header, section

from trustworthy_evals.agent_eval import (
    AUTH_BYPASS_SPEC_YAML,
    NumericMatch,
    ToolCall,
    Transcript,
    auth_bypass_task,
    grade_task,
    make_flaky_agent,
    run_trials,
    trajectory_match,
)
from trustworthy_evals.metrics import pass_at_k, pass_hat_k


def main() -> None:
    header("Part 9 - Agent evaluation: outcomes, partial credit, and rates")

    section("A concrete task spec (fix-auth-bypass-01)")
    print("  " + AUTH_BYPASS_SPEC_YAML.replace("\n", "\n  ").rstrip())

    task = auth_bypass_task()
    section("Partial credit: graded by outcome, not by 'did it fail'")
    full = Transcript(
        final_output="Rejected empty/null passwords; auth_blocked logged.",
        tool_calls=[ToolCall("edit_file"), ToolCall("run_tests")],
        final_state={"security_logs": {"event_type": "auth_blocked"}, "quality": 0.85},
        tests_passed={"test_empty_pw_rejected.py": True, "test_null_pw_rejected.py": True},
    )
    partial = Transcript(
        final_output="edited the file",
        tool_calls=[ToolCall("edit_file"), ToolCall("run_tests")],
        final_state={"security_logs": {}, "quality": 0.6},
        tests_passed={"test_empty_pw_rejected.py": True, "test_null_pw_rejected.py": False},
    )
    nothing = Transcript(final_output="", final_state={"quality": 0.2})
    for label, tr in [("full success", full), ("partial work", partial), ("did nothing", nothing)]:
        r = grade_task(task, tr)
        print(f"  {label:13s}: passed={str(r.passed):5s} score={r.score:.2f}  | {r.detail()}")

    section("Outcome over trajectory (grade the path only where you care)")
    print(f"  strict   ['a','b','c'] vs ['a','b','c']        : {trajectory_match(list('abc'), list('abc'), 'strict')}")
    print(f"  unordered['c','a','b'] vs ['a','b','c']        : {trajectory_match(list('cab'), list('abc'), 'unordered')}")
    print(f"  superset ['a','b','c','d'] >= reference ['a','c']: {trajectory_match(list('abcd'), list('ac'), 'superset')}")

    section("Non-determinism: report a rate (pass@k rises, pass^k falls)")
    report = run_trials(task, make_flaky_agent(0.75), k=10)
    print(f"  per-trial success rate : {report.observed_rate:.2f}")
    for k in (1, 3, 10):
        print(f"  k={k:<2d}  pass@k={pass_at_k(report.observed_rate, k):.2f}   pass^k={pass_hat_k(report.observed_rate, k):.2f}")
    print("  -> at k=10 they tell opposite stories: 'at least one' ~1.0 vs 'every time' ~0.06.")

    section("Reward hacking via a rigid grader")
    tr = Transcript(final_output="The computed key is 96.12")
    print(f"  rigid  NumericMatch(96.124991, tol=0)    -> passed={NumericMatch(96.124991, tol=0.0).grade(tr).passed}  (rejects a correct answer!)")
    print(f"  tolerant NumericMatch(96.124991, tol=.01)-> passed={NumericMatch(96.124991, tol=0.01).grade(tr).passed}")
    print("  A 0% pass@100 with a frontier model usually means a broken task/grader -- read the transcripts.")

    print("\n  Lesson: prefer deterministic graders, add an LLM rubric for nuance, validate with")
    print("  humans; grade end-state; partial-credit multi-step tasks; run isolated trials; report a rate.")


if __name__ == "__main__":
    main()
