import math

import pytest

from trustworthy_evals.metrics import (
    agreement_rate,
    cohens_kappa,
    normalize,
    pass_at_k,
    pass_at_k_estimate,
    pass_hat_k,
    pearson,
    success_rate,
)


def test_pearson_perfect_and_anti():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_zero_variance_is_safe():
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0


def test_normalize():
    assert normalize(3, 1, 4) == pytest.approx(2 / 3)
    assert normalize(4, 1, 5) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        normalize(1, 2, 2)


def test_agreement_and_kappa():
    a = ["x", "x", "y", "y"]
    b = ["x", "x", "y", "y"]
    assert agreement_rate(a, b) == 1.0
    assert cohens_kappa(a, b) == pytest.approx(1.0)
    # Chance-level labeling -> kappa near 0.
    c = ["x", "y", "x", "y"]
    d = ["x", "x", "y", "y"]
    assert cohens_kappa(c, d) == pytest.approx(0.0, abs=1e-9)


def test_pass_at_k_vs_pass_hat_k_opposite_directions():
    p = 0.5
    # pass@k rises with k, pass^k falls with k.
    assert pass_at_k(p, 1) < pass_at_k(p, 10)
    assert pass_hat_k(p, 1) > pass_hat_k(p, 10)
    # The tutorial's worked example.
    assert pass_hat_k(0.75, 3) == pytest.approx(0.421875)
    # Identical at k=1.
    assert pass_at_k(p, 1) == pass_hat_k(p, 1) == p


def test_pass_at_k_estimate_matches_intuition():
    # All trials succeed -> certain.
    assert pass_at_k_estimate(n=10, c=10, k=5) == pytest.approx(1.0)
    # No successes -> zero.
    assert pass_at_k_estimate(n=10, c=0, k=5) == 0.0
    # Monotonic in k.
    assert pass_at_k_estimate(20, 5, 1) < pass_at_k_estimate(20, 5, 5)


def test_success_rate():
    assert success_rate([True, False, True, True]) == 0.75


def test_input_validation():
    with pytest.raises(ValueError):
        pearson([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        pass_at_k(1.2, 3)
    with pytest.raises(ValueError):
        pass_hat_k(0.5, 0)
