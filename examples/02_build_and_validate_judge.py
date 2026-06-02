"""Part 2 - Building the judge, and validating it before you trust it.

Never ship a judge you have not measured against humans. Collect a small
human-labeled set, compute the two-annotator agreement (your ceiling), then
show that a few prompt changes move a naive 0-10 judge from ~0.57 to ~0.84.
"""

import statistics

from _bootstrap import header, section

from trustworthy_evals.judge import extract_score, simulate_validation_study
from trustworthy_evals.prompts import IMPROVED_JUDGE_PROMPT, NAIVE_JUDGE_PROMPT


def main() -> None:
    header("Part 2 - Build the judge, validate it first")

    section("Robust score parsing (anchor on the marker, grab the number)")
    for raw in ["Evaluation: thorough and correct.\nTotal rating: 4",
                "I'd say Total rating: 3.5 honestly",
                "completely unparseable prose"]:
        print(f"  extract_score({raw[:42]!r:44}) -> {extract_score(raw)}")

    section("Validation study (averaged over 8 seeds)")
    inter, naive, improved = [], [], []
    for s in range(8):
        study = simulate_validation_study(seed=s)
        sub = study.agreement_subset()
        inter.append(study.inter_rater())
        naive.append(study.naive_correlation(sub))
        improved.append(study.improved_correlation(sub))
    print(f"  two-human inter-rater correlation : {statistics.mean(inter):.3f}   (the CEILING; cookbook ~0.563)")
    print(f"  naive 0-10 float judge            : {statistics.mean(naive):.3f}   (barely above ceiling; ~0.567)")
    print(f"  improved anchored 1-4 judge       : {statistics.mean(improved):.3f}   (cookbook ~0.843)")

    section("What changed between the two prompts")
    print("  naive prompt    :", NAIVE_JUDGE_PROMPT.strip().splitlines()[1].strip())
    print("  improved prompt : reasoning-before-score + anchored integer scale + parseable marker")
    print("\n  Lesson: prompt-tweaking moves correlation dramatically, but you can only SEE")
    print("  that because you validated against humans. The human ceiling caps how high you go.")


if __name__ == "__main__":
    main()
