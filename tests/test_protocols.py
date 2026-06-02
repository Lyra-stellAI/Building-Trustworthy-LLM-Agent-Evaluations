import statistics

from trustworthy_evals.llm import DISTRACTORS
from trustworthy_evals.protocols import (
    leaderboard_hacking_demo,
    summarize_distracted,
    tie_recognition_experiment,
)


def test_pairwise_flips_far_more_than_absolute():
    results = summarize_distracted()
    pw = statistics.mean(r.pairwise_flip_rate for r in results.values())
    ab = statistics.mean(r.absolute_flip_rate for r in results.values())
    # The COLM 2025 headline: ~35% pairwise vs ~9% absolute.
    assert 0.25 <= pw <= 0.48, pw
    assert ab <= 0.15, ab
    assert pw > 2 * ab


def test_every_distractor_flips_pairwise_more_than_absolute():
    results = summarize_distracted()
    assert set(results) == set(DISTRACTORS)
    for d, r in results.items():
        assert r.pairwise_flip_rate > r.absolute_flip_rate, d


def test_absolute_recognizes_ties_pairwise_does_not():
    identical, tie = tie_recognition_experiment()
    assert 0.80 <= identical <= 0.95, identical  # paper: 84-93%
    assert tie <= 0.12, tie  # paper: 2-7%
    assert identical > tie


def test_leaderboard_hacking_moves_pairwise_not_absolute():
    hack = leaderboard_hacking_demo()
    # Hacked models climb the pairwise board...
    gains = [hack.pairwise_rank_change(m) for m in hack.hacked]
    assert sum(gains) > 0
    assert any(g > 0 for g in gains)
    # ...while the absolute board is unmoved by content-free assertiveness.
    assert hack.absolute_before == hack.absolute_after
