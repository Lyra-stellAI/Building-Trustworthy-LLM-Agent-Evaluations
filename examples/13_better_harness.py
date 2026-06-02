"""Part 13.2 - better-harness: an eval-driven harness optimizer.

One agent improves another agent's harness, keeping a change only if the evals
say it generalized. The held-out split is the whole point: a change that
memorizes the visible train cases without generalizing fails the hidden holdout
and is rejected -- the Part 3 mechanism-design thesis made literal.
"""

from _bootstrap import header, section

from trustworthy_evals.better_harness import optimize


def main() -> None:
    header("Part 13.2 - better-harness: optimize against generalization, not the visible set")

    run = optimize()

    section("Decision log (the outer agent sees only TRAIN failures)")
    print(f"  {'it':>2s} {'verdict':7s} {'kind':13s} {'candidate':32s} {'train':>5s} {'hold':>5s}  reason")
    for d in run.decisions:
        verdict = "KEEP" if d.accepted else "drop"
        print(f"  {d.iteration:>2d} {verdict:7s} {d.kind:13s} {d.candidate:32s} {d.train_pass:>5d} {d.holdout_pass:>5d}  {d.reason}")

    section("Result")
    print(f"  train     : {run.baseline_train} -> {run.final_train}")
    print(f"  holdout   : {run.baseline_holdout} -> {run.final_holdout}   (private; the generalization guard)")
    print(f"  scorecard : {run.baseline_scorecard} -> {run.final_scorecard}   (untouched during the loop -> the honest final report)")
    print(f"  accepted generalizing edits: {len(run.accepted())}; rejected overfitting hacks: {len(run.rejected_overfits())}")

    print("\n  Lesson: the overfitting 'memorize' candidate always maxes TRAIN -- and is rejected")
    print("  every time because it regresses the hidden HOLDOUT. The structure (not the prompt)")
    print("  guarantees the only way to score is to genuinely improve the harness.")


if __name__ == "__main__":
    main()
