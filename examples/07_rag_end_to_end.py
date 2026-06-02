"""Part 7 - Putting it together: RAG evaluation end to end.

A RAG pipeline has many knobs; tuning them is pointless if you can't measure the
effect. Sweep chunk size / rerank / reader, score each with an absolute anchored
judge, and watch the empirical conclusion emerge: no single recipe, but
chunk-size is cheap and high-impact.
"""

from _bootstrap import header, section

from trustworthy_evals.prompts import RAG_EVAL_PROMPT
from trustworthy_evals.rag_eval import sweep_configs


def main() -> None:
    header("Part 7 - RAG evaluation: sweep the knobs, measure the number")

    section("The judge: absolute, anchored 1-5, reference answer + [RESULT] marker")
    print("  " + RAG_EVAL_PROMPT.strip().splitlines()[0])
    print("  ... (anchored 1-5 rubric; GPT-class judge, a different family than the reader)")

    section("Config sweep (normalized answer-correctness, higher is better)")
    print(f"  {'configuration':50s} {'score':>6s} {'hit_rate':>9s}")
    for r in sweep_configs():
        print(f"  {r.config.label():50s} {r.score:6.3f} {r.hit_rate:9.0%}")

    print("\n  Lesson: chunk size is the big lever here (retrieval hit-rate drives everything);")
    print("  rerank does nothing in THIS system; the strong reader helps. Which tweak helps is")
    print("  system-specific -- the pipeline exists so you stop guessing and see what moves the number.")


if __name__ == "__main__":
    main()
