from trustworthy_evals.agent_eval import StateCheck, Task, grade_task
from trustworthy_evals.deep_agent import Coverage, LLMWiki, LogContains, WikiGroundedness


def test_ingest_changes_state_and_pushes_revision():
    wiki = LLMWiki()
    res = wiki.ingest("topic", ["claim one", "claim two"])
    st = res.transcript.final_state
    assert st["index_refreshed"] is True
    assert st["revision_pushed"] is True
    assert st["wrote"] is True
    assert "wiki/pages/topic.md" in st["pages_written"]


def test_review_decline_skips_all_writes():
    wiki = LLMWiki()
    before = dict(wiki.fs)
    res = wiki.ingest("draft", ["unverified"], review=True, confirm=False)
    assert res.transcript.final_state["wrote"] is False
    # No page was written and no revision pushed.
    assert "wiki/pages/draft.md" not in wiki.fs
    assert wiki.revision == 0
    # The log records the skip.
    assert "ingest.review | outcome=skipped" in wiki.log_text()


def test_query_is_grounded_and_logged():
    wiki = LLMWiki()
    wiki.ingest("llm-judges", ["evaluating text is easier than generating it"])
    res = wiki.query(
        "Why does it work?",
        claims=["evaluating text is easier than generating it"],
        cite=["wiki/pages/llm-judges.md"],
    )
    assert WikiGroundedness().grade(res.transcript).passed
    assert "query.apply | outcome=filed" in res.transcript.final_state["log"]


def test_groundedness_catches_hallucination():
    wiki = LLMWiki()
    wiki.ingest("llm-judges", ["evaluating text is easier than generating it"])
    res = wiki.query(
        "Why does it work?",
        claims=["jupiter is the largest planet in the solar system"],
        cite=["wiki/pages/llm-judges.md"],
    )
    result = WikiGroundedness().grade(res.transcript)
    assert not result.passed
    assert result.score < 0.5


def test_bespoke_grader_task_for_query():
    wiki = LLMWiki()
    wiki.ingest("llm-judges", ["strong judges reach about 80 percent agreement with humans"])
    res = wiki.query(
        "How well do judges agree?",
        claims=["strong judges reach about 80 percent agreement with humans"],
        cite=["wiki/pages/llm-judges.md"],
    )
    task = Task("wiki-query", "grounded + covered + filed", [
        WikiGroundedness(threshold=0.5, required=True),
        Coverage(required=["wiki/index.md", "wiki/pages/llm-judges.md"]),
        LogContains("query.apply | outcome=filed"),
    ], mode="hybrid", pass_threshold=0.6)
    assert grade_task(task, res.transcript).passed


def test_lint_finds_orphans():
    wiki = LLMWiki()
    # Manually drop a page into the fs without indexing it -> orphan.
    wiki.fs["wiki/pages/orphan.md"] = "# orphan"
    res = wiki.lint()
    assert "wiki/pages/orphan.md" in res.transcript.final_state["orphans"]
