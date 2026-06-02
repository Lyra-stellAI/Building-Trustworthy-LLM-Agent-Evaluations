import pytest

from trustworthy_evals.harness import (
    DEEP_AGENT_COMPONENTS,
    DEEP_AGENT_MIDDLEWARE,
    STACK_LAYERS,
    Harness,
    Surface,
    boundary_enforcement,
    harness_engineering_demo,
    skills_ablation_demo,
)


def test_harness_engineering_reproduces_langchain_result():
    g = harness_engineering_demo()
    assert g.baseline_score == pytest.approx(52.8)
    assert g.improved_score == pytest.approx(66.5)
    assert g.delta == pytest.approx(13.7, abs=1e-6)


def test_skills_ablation_reproduces_finding():
    s = skills_ablation_demo()
    assert s.baseline_score == pytest.approx(9.0)
    assert s.improved_score == pytest.approx(82.0)
    assert s.improved_score > 8 * s.baseline_score  # curated skills are dramatic


def test_harness_only_changes_with_model_fixed():
    # The score moves purely from surfaces; the base (model) never changes.
    h = Harness(base=50.0)
    assert h.score() == 50.0
    h2 = h.with_surface(Surface("prompt_v2", "prompt", 10.0), quality=1.0)
    assert h2.base == h.base  # model held fixed
    assert h2.score() == 60.0
    # Quality scales the contribution.
    h3 = h.with_surface(Surface("prompt_v2", "prompt", 10.0), quality=0.5)
    assert h3.score() == 55.0


def test_score_is_clamped_to_scale():
    h = Harness(base=95.0).with_surface(Surface("big", "skills", 50.0))
    assert h.score() == 100.0


def test_boundary_enforcement_sandbox_vs_prompt():
    # A sandbox-enforced boundary holds regardless of the model.
    assert boundary_enforcement(sandbox_enforced=True, model_complies=False)
    assert boundary_enforcement(sandbox_enforced=True, model_complies=True)
    # A prompt-policed boundary holds only if the model chooses to comply.
    assert not boundary_enforcement(sandbox_enforced=False, model_complies=False)
    assert boundary_enforcement(sandbox_enforced=False, model_complies=True)


def test_stack_and_deep_agent_docs_present():
    assert set(STACK_LAYERS) == {"LangChain", "LangGraph", "Deep Agents"}
    for key in ["planning", "virtual_filesystem", "sub_agents", "middleware"]:
        assert key in DEEP_AGENT_COMPONENTS
    assert "human-in-the-loop" in DEEP_AGENT_MIDDLEWARE
