import statistics

import pytest

from trustworthy_evals.judge import Judge, extract_score, simulate_validation_study
from trustworthy_evals.llm import ScriptedLLM


def test_extract_score_variants():
    assert extract_score("Evaluation: good.\nTotal rating: 4") == 4.0
    assert extract_score("blah Total rating: 3.5 of 4") == 3.5
    assert extract_score("no marker, score 7 here") == 7.0
    assert extract_score("nothing numeric at all") is None


def test_extract_score_uses_last_marker():
    text = "Total rating: ignore\nmore\nTotal rating: 2"
    assert extract_score(text) == 2.0


def test_judge_normalizes_to_unit_interval():
    judge = Judge(ScriptedLLM(["Evaluation: ok\nTotal rating: 3"]))
    assert judge.score(question="q", answer="a") == pytest.approx(2 / 3)


def test_naive_judge_scale():
    judge = Judge.naive(ScriptedLLM(["Total rating: 5"]))
    assert judge.score(question="q", answer="a") == pytest.approx(0.5)


def test_validation_study_reproduces_cookbook_story():
    # Average over seeds: improved judge clearly beats the naive one, and the
    # naive judge is no better than the human baseline.
    inter, naive, improved = [], [], []
    for s in range(8):
        study = simulate_validation_study(seed=s)
        sub = study.agreement_subset()
        inter.append(study.inter_rater())
        naive.append(study.naive_correlation(sub))
        improved.append(study.improved_correlation(sub))
    mi, mn, mim = statistics.mean(inter), statistics.mean(naive), statistics.mean(improved)
    assert 0.45 <= mi <= 0.70, mi  # ~0.563 ceiling
    assert 0.45 <= mn <= 0.70, mn  # ~0.567 naive
    assert mim >= 0.78, mim  # ~0.843 improved
    assert mim > mn + 0.15  # the prompt fix is a large, real gain
