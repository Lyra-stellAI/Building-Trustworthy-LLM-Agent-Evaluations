"""Part 1 - Building a synthetic evaluation dataset.

Generate QA pairs from a corpus, then run critique agents (groundedness,
relevance, standalone-ness) and keep only pairs scoring >= 4 on all three.
Expect to discard roughly half -- so generate twice what you need.
"""

from _bootstrap import header, section

from trustworthy_evals.datasets import candidate_qa_pairs, filter_eval_set
from trustworthy_evals.prompts import CRITIQUE_CRITERIA


def main() -> None:
    header("Part 1 - Synthetic evaluation dataset + critique-agent filtering")

    section("The three critique criteria")
    for name, question in CRITIQUE_CRITERIA.items():
        print(f"  {name:12s}: {question}")

    candidates = candidate_qa_pairs()
    report = filter_eval_set(candidates)

    section("Critique scores per candidate (G=groundedness R=relevance S=standalone)")
    for qa in candidates:
        verdict = "KEEP" if qa.passes(4) else "drop"
        flaws = ",".join(sorted(qa.flaws)) or "-"
        print(f"  [{verdict}] G{qa.groundedness} R{qa.relevance} S{qa.standalone}  "
              f"flaw={flaws:18s} {qa.question[:48]}")

    section("Result")
    print(f"  generated : {report.total}")
    print(f"  survivors : {report.kept}  ({report.keep_rate:.0%} kept)")
    print(f"  discarded : {report.rejected}")
    print("\n  Lesson: filters are aggressive (~half discarded). Aim for 100+ survivors,")
    print("  so generate 200+. Each rejected pair failed exactly the criterion its flaw targets.")


if __name__ == "__main__":
    main()
