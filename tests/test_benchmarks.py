from trustworthy_evals.benchmarks import BENCHMARKS, LEADERBOARD_CAUTIONS, by_category


def test_registry_covers_the_canonical_benchmarks():
    names = " ".join(b.name for b in BENCHMARKS)
    for expected in ["SWE-bench", "Terminal-Bench", "tau-bench", "BFCL", "GAIA", "WebArena", "OSWorld", "BrowseComp"]:
        assert expected in names


def test_by_category_filters():
    coding = by_category("Coding")
    assert coding
    assert all("coding" in b.category.lower() for b in coding)


def test_two_cautions_present():
    text = " ".join(LEADERBOARD_CAUTIONS).lower()
    assert "saturation" in text
    assert "harness" in text
