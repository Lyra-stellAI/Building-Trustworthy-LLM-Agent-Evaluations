"""Part 12 - Open-source agent frameworks: LangChain, LangGraph, Deep Agents.

agent = model + harness. Most of an agent's behavior, and most of your headroom
to improve it, lives in the harness (prompt, tools, skills, context management,
middleware) -- not the frozen weights.
"""

from _bootstrap import header, section

from trustworthy_evals.harness import (
    DEEP_AGENT_COMPONENTS,
    DEEP_AGENT_MIDDLEWARE,
    STACK_LAYERS,
    boundary_enforcement,
    harness_engineering_demo,
    skills_ablation_demo,
)


def main() -> None:
    header("Part 12 - agent = model + harness")

    section("The three layers of the open-source stack")
    for layer, role in STACK_LAYERS.items():
        print(f"  {layer:12s}: {role}")

    section("Deep Agents - the batteries-included harness")
    for name, desc in DEEP_AGENT_COMPONENTS.items():
        print(f"  {name:18s}: {desc}")
    print(f"  default middleware stack (in order): {' -> '.join(DEEP_AGENT_MIDDLEWARE)}")

    section("Harness engineering: change ONLY the harness, model held fixed")
    g = harness_engineering_demo()
    print(f"  Terminal-Bench 2.0: {g.baseline_score} -> {g.improved_score}  (+{g.delta:.1f} points)")
    print("  Same weights (gpt-5.2-codex); roughly rank 30 -> top 5, from harness changes alone.")

    section("Curated skills are a harness surface with outsized impact")
    s = skills_ablation_demo()
    print(f"  task completion: {s.baseline_score:.0f}% without curated skills -> {s.improved_score:.0f}% with")

    section("Trust the LLM, enforce at the tool / sandbox level (ties to Part 9)")
    print(f"  sandbox-enforced limit, model tries to defy : holds = {boundary_enforcement(sandbox_enforced=True, model_complies=False)}")
    print(f"  prompt-policed limit, model tries to defy   : holds = {boundary_enforcement(sandbox_enforced=False, model_complies=False)}")

    print("\n  Lesson: the harness is the unit of optimization. An eval suite tells you whether")
    print("  turning a knob (prompt, tool, skill, middleware) helped or just moved the failure.")


if __name__ == "__main__":
    main()
