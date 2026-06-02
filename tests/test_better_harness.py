from trustworthy_evals.better_harness import HarnessConfig, default_splits, optimize


def test_optimizer_improves_holdout_and_scorecard():
    run = optimize()
    # The loop climbs train to full...
    assert run.final_train >= run.baseline_train
    assert run.final_train == len(default_splits().train)
    # ...and crucially improves the PRIVATE holdout and scorecard it optimizes
    # against generalization for.
    assert run.final_holdout > run.baseline_holdout
    assert run.final_scorecard >= run.baseline_scorecard
    assert run.holdout_gain > 0


def test_overfitting_hack_is_always_rejected():
    run = optimize()
    overfits = [d for d in run.decisions if d.kind == "overfitting"]
    assert overfits  # the hack is considered...
    assert all(not d.accepted for d in overfits)  # ...and never accepted.
    # At least one is rejected specifically for regressing the holdout.
    assert any("holdout regressed" in d.reason for d in overfits)


def test_accepted_changes_are_generalizing():
    run = optimize()
    accepted = run.accepted()
    assert accepted
    assert all(d.kind == "generalizing" for d in accepted)


def test_memorize_maxes_train_but_loses_holdout():
    # The overfit hack hard-codes the visible train failures (max train) while
    # degrading the general policy, so the held-out split drops.
    split = default_splits()
    base = HarnessConfig()
    failures = [c.id for c in split.train if base.capability(c) < c.difficulty]
    hacked = base.memorize(failures)
    assert hacked.n_pass(split.train) > base.n_pass(split.train)
    assert hacked.n_pass(split.holdout) < base.n_pass(split.holdout)
