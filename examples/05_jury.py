"""Part 5 - Panels and juries: ablate across judges, then vote.

A single judge is a single point of failure: judge choice silently determines
results, and a judge over-rates its own family. A jury of small, diverse models
(PoLL) agrees with humans better, shows less intra-model bias, and costs far less.
"""

from _bootstrap import header, section

from trustworthy_evals.jury import judge_ablation, panel_vs_single_demo
from trustworthy_evals.llm import Response, SimulatedJudge


def main() -> None:
    header("Part 5 - Panel of LLM evaluators (jury) beats one big judge")

    section("Single shared-lineage judge vs. a diverse panel (Cohen's kappa vs humans)")
    comp = panel_vs_single_demo()
    print(f"  single large judge (shares lineage with the system) : kappa = {comp.single_kappa:.3f}")
    print(f"  panel of 3 small judges from disjoint families       : kappa = {comp.panel_kappa:.3f}")
    print(f"  panel cost vs single                                 : {comp.cost_ratio:.1f}x cheaper")
    print("  -> self-preference inflates the WHOLE eval set for the single judge; the panel dilutes it.")

    section("Judge ablation: does 'variant B beats variant A' survive a change of judge?")
    variant_a = Response(text="A", quality=0.55)
    variant_b = Response(text="B", quality=0.62)
    judges = [SimulatedJudge(family=f, seed=i) for i, f in enumerate(["gpt", "claude", "qwen"])]
    robust, per_judge = judge_ablation(
        judges, lambda j: j.score_absolute(variant_b) >= j.score_absolute(variant_a)
    )
    for name, verdict in per_judge.items():
        print(f"    {name:12s}: B>=A ? {verdict}")
    print(f"  robust conclusion (holds under every judge): {robust}")

    print("\n  Lesson: use an odd number of diverse judges; majority vote for binary verdicts,")
    print("  average pooling for graded scores; route strong disagreement to a human. And always")
    print("  ablate across >=2 families -- if your conclusion flips with the judge, it's an artifact.")


if __name__ == "__main__":
    main()
