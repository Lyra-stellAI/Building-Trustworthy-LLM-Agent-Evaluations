"""Part 6 - Calibration: turning a judge into a trustworthy instrument.

Prompt iteration alone won't close the gap. The loop does: collect human
corrections -> build few-shot examples -> track agreement over time. Watch
judge-human agreement climb as the judge learns a blind-spot it started by
ignoring.
"""

from _bootstrap import header, section

from trustworthy_evals.calibration import (
    DATA_FLYWHEEL,
    EvalTask,
    choose_evaluator,
    run_calibration_loop,
)


def main() -> None:
    header("Part 6 - The calibration loop")

    section("Agreement rises as corrections become few-shot examples")
    rounds = run_calibration_loop()
    for r in rounds:
        bar = "#" * int(r.agreement * 40)
        print(f"  round {r.round_index}: feature_coverage={r.feature_coverage:.2f}  "
              f"agreement={r.agreement:.3f}  corrections_needed={r.n_corrections:3d}  {bar}")
    print(f"  net gain over the loop: {rounds[-1].agreement - rounds[0].agreement:+.3f}")

    section("Where an LLM judge fits in the stack (use the cheapest capable tool)")
    cases = [
        ("format/length check", EvalTask(mechanical=True)),
        ("has a gold reference", EvalTask(has_reference=True)),
        ("helpfulness / tone", EvalTask(nuanced=True)),
        ("high-stakes medical", EvalTask(high_stakes=True, nuanced=True)),
    ]
    for desc, task in cases:
        print(f"  {desc:22s} -> {choose_evaluator(task)}")

    section("The data flywheel")
    print(f"  {DATA_FLYWHEEL}")

    print("\n  Lesson: prompt-tweaking alone is not enough. The systematic loop -- corrections,")
    print("  few-shot, tracked agreement -- is what makes a judge trustworthy for shipping.")


if __name__ == "__main__":
    main()
