from trustworthy_evals.biases import (
    BIAS_TABLE,
    position_bias_demo,
    self_enhancement_demo,
    verbosity_bias_demo,
)


def test_position_bias_present_and_caught_by_both_orders():
    demo = position_bias_demo()
    # Single-order strongly favors slot A (biased > 0.5 win rate).
    assert demo.biased > 0.2
    # Running both orders flags the unstable pairs instead of trusting them, so
    # very few equal-quality pairs are reported "consistent".
    assert demo.mitigated < 0  # consistent-rate - 0.5 is well below 0


def test_verbosity_bias_shrinks_under_mitigation():
    demo = verbosity_bias_demo()
    assert demo.biased > 0.1  # longer answers score higher at equal quality
    assert demo.shrunk
    assert abs(demo.mitigated) < 0.05  # "ignore length" instruction removes it


def test_self_enhancement_shrinks_with_neutral_judge():
    demo = self_enhancement_demo()
    assert demo.biased > 0.1  # own family scores higher
    assert demo.shrunk
    assert abs(demo.mitigated) < 0.05  # third-family judge is neutral


def test_bias_table_is_complete():
    names = [row[0] for row in BIAS_TABLE]
    assert "Position bias" in names
    assert "Distracted evaluation" in names
    assert len(BIAS_TABLE) == 6
    assert all(len(row) == 3 for row in BIAS_TABLE)
