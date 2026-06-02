"""Part 4 - Known biases and how to mitigate them.

A checklist to design against. Each demo shows the bias with the default judge,
then shows the mitigation shrinking it. Instruction helps but is not a complete
fix -- which is why calibration (Part 6) is non-negotiable.
"""

from _bootstrap import header, section

from trustworthy_evals.biases import (
    BIAS_TABLE,
    position_bias_demo,
    self_enhancement_demo,
    verbosity_bias_demo,
)


def main() -> None:
    header("Part 4 - Known biases and mitigations")

    section("The bias checklist")
    for bias, what, mitigation in BIAS_TABLE:
        print(f"  * {bias}")
        print(f"      what : {what}")
        print(f"      fix  : {mitigation}")

    section("Position bias (pairwise, equal-quality pairs)")
    p = position_bias_demo()
    print(f"  {p.note}")
    print(f"  -> single-order judging is biased; both-orders flags the unstable pairs instead.")

    section("Verbosity / length bias (absolute, equal-quality long vs short)")
    v = verbosity_bias_demo()
    print(f"  mean score advantage for the longer answer : {v.biased:+.3f}")
    print(f"  after 'ignore length' instruction          : {v.mitigated:+.3f}")

    section("Self-enhancement bias (judge rates its own family higher)")
    s = self_enhancement_demo()
    print(f"  own-family score advantage (same-family judge) : {s.biased:+.3f}")
    print(f"  with a neutral third-family judge              : {s.mitigated:+.3f}")

    print("\n  Lesson: tell the judge to ignore length, position, and names -- it helps -- but")
    print("  instruction alone never fully removes bias. Calibrate against your own domain.")


if __name__ == "__main__":
    main()
