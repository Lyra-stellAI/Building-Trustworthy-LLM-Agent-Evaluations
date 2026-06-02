from trustworthy_evals.datasets import (
    candidate_qa_pairs,
    critique_qa,
    filter_eval_set,
    generate_qa,
)
from trustworthy_evals.llm import ScriptedLLM


def test_clean_pairs_pass_flawed_pairs_fail():
    report = filter_eval_set(candidate_qa_pairs())
    # Every survivor scores >= 4 on all three criteria.
    for qa in report.survivors:
        assert qa.passes(4)
    # Every rejected pair carries at least one flaw.
    for qa in report.rejected_pairs:
        assert qa.flaws


def test_filter_discards_roughly_half():
    report = filter_eval_set(candidate_qa_pairs())
    assert 0.35 <= report.keep_rate <= 0.65, report.keep_rate
    assert report.kept + report.rejected == report.total


def test_critique_targets_the_right_flaw():
    (clean, *_), = [candidate_qa_pairs()[:1]]
    # A clean pair scores high on every criterion.
    for c in ("groundedness", "relevance", "standalone"):
        assert critique_qa(clean, c) >= 4


def test_generate_qa_real_path_parsing():
    llm = ScriptedLLM(["Factoid question: What is X?\nAnswer: It is Y."])
    qa = generate_qa("some context", llm)
    assert qa.question == "What is X?"
    assert qa.answer == "It is Y."


def test_critique_real_path_uses_marker():
    llm = ScriptedLLM(["Evaluation: solid and answerable.\nTotal rating: 5"])
    assert critique_qa(candidate_qa_pairs()[0], "groundedness", llm=llm) == 5
