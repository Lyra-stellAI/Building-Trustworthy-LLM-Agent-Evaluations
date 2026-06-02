"""Part 13.1 - The LLM Wiki: a Deep Agent worth evaluating.

A Deep Agent that builds and maintains a knowledge base across ingest / query /
lint modes. It touches most of the Part 9 agent-eval surface, so we grade it with
bespoke, per-case graders: state checks (did the index refresh, did a revision
push, did a declined review skip writes?), groundedness of cited answers, and a
code-based grader over the machine-parseable log.
"""

from _bootstrap import header, section

from trustworthy_evals.agent_eval import StateCheck, Task, grade_task
from trustworthy_evals.deep_agent import Coverage, LLMWiki, LogContains, WikiGroundedness


def main() -> None:
    header("Part 13.1 - Evaluating the LLM Wiki Deep Agent")
    wiki = LLMWiki()

    section("ingest: expand a source into a wiki page (state-change work)")
    res = wiki.ingest("llm-judges", [
        "evaluating text is easier than generating it",
        "strong judges reach about 80 percent agreement with humans",
    ])
    print(f"  {res.output}")
    ingest_task = Task("wiki-ingest", "ingest refreshes index and pushes a revision", [
        StateCheck({"index_refreshed": True}, required=True),
        StateCheck({"revision_pushed": True}, required=True),
    ], mode="binary")
    r = grade_task(ingest_task, res.transcript)
    print(f"  grade: passed={r.passed} | {r.detail()}   <- grade the OUTCOME, not the chat reply")

    section("ingest --review then DECLINE: must skip all writes (human-in-the-loop)")
    declined = wiki.ingest("draft-topic", ["unverified claim"], review=True, confirm=False)
    skip_task = Task("wiki-decline", "a declined review writes nothing", [
        StateCheck({"wrote": False}, required=True),
    ], mode="binary")
    r = grade_task(skip_task, declined.transcript)
    print(f"  {declined.output}")
    print(f"  grade: passed={r.passed} | {r.detail()}")

    section("query: answer with citations; grade groundedness + coverage + the log")
    grounded = wiki.query(
        "Why does LLM-as-judge work?",
        claims=["evaluating text is easier than generating it"],
        cite=["wiki/pages/llm-judges.md"],
    )
    query_task = Task("wiki-query", "grounded, well-sourced, filed answer", [
        WikiGroundedness(threshold=0.5, required=True),
        Coverage(required=["wiki/index.md", "wiki/pages/llm-judges.md"]),
        LogContains("query.apply | outcome=filed"),
    ], mode="hybrid", pass_threshold=0.6)
    r = grade_task(query_task, grounded.transcript)
    print(f"  answer: {grounded.output}")
    print(f"  grade: passed={r.passed} | {r.detail()}")

    section("the groundedness grader catches a hallucinated answer")
    halluc = wiki.query(
        "Why does LLM-as-judge work?",
        claims=["the moon orbits the earth roughly every 27 days"],
        cite=["wiki/pages/llm-judges.md"],
    )
    score = WikiGroundedness().grade(halluc.transcript)
    print(f"  hallucinated answer groundedness: {score.score:.2f} (passed={score.passed})")

    section("lint: find orphan pages")
    print(f"  {wiki.lint().output}")

    print("\n  Lesson: agent evals are bespoke per case. 'review skipped writes on decline' and")
    print("  'the answer was grounded in cited pages' are different assertions on different artifacts.")


if __name__ == "__main__":
    main()
