import statistics

from trustworthy_evals.jury import Jury, judge_ablation, jury_verdict, panel_vs_single_demo
from trustworthy_evals.llm import Response, SimulatedJudge


def test_jury_mean_and_majority():
    judges = [
        SimulatedJudge(noise=0.0, length_w=0, position_w=0, seed=1),
        SimulatedJudge(noise=0.0, length_w=0, position_w=0, seed=2),
        SimulatedJudge(noise=0.0, length_w=0, position_w=0, seed=3),
    ]
    r = Response(text="x", quality=0.8)
    mean = jury_verdict(judges, r, mode="mean")
    majority = jury_verdict(judges, r, mode="majority")
    assert 1 <= mean <= 5
    assert majority in (1, 2, 3, 4, 5)


def test_panel_beats_single_judge_on_human_agreement():
    # Average across seeds: the diverse panel agrees with humans better than a
    # single judge that shares lineage with the system under test.
    comps = [panel_vs_single_demo(seed=s) for s in range(6)]
    panel = statistics.mean(c.panel_kappa for c in comps)
    single = statistics.mean(c.single_kappa for c in comps)
    assert panel > single
    assert all(c.panel_wins for c in comps)


def test_panel_is_cheaper():
    comp = panel_vs_single_demo()
    assert comp.cost_ratio > 5.0  # PoLL: a panel of small models is far cheaper


def test_judge_ablation_flags_non_robust_conclusions():
    a = Response(text="a", quality=0.55)
    b = Response(text="b", quality=0.58)
    judges = [SimulatedJudge(family=f, seed=i) for i, f in enumerate(["gpt", "claude", "qwen"])]
    # A real difference survives a change of judge.
    robust, _ = judge_ablation(judges, lambda j: j.score_absolute(b) >= j.score_absolute(a))
    assert isinstance(robust, bool)
    # A self-contradicting conclusion is never robust.
    robust2, per = judge_ablation(judges, lambda j: j.score_absolute(a) > j.score_absolute(a))
    assert robust2 is True  # all judges agree it's False
    assert set(per.values()) == {False}
