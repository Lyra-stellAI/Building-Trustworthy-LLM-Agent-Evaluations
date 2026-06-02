"""Part 10 & 11 - The platform landscape and agent benchmarks worth knowing.

Public benchmarks won't tell you whether YOUR agent works -- build your own task
bank from real failures for that -- but they are the shared language of the
field. Two cautions apply to every leaderboard.
"""

from _bootstrap import header, section

from trustworthy_evals.agent_eval import AGENT_PLAYBOOKS
from trustworthy_evals.benchmarks import BENCHMARKS, LEADERBOARD_CAUTIONS


def main() -> None:
    header("Part 11 - Agent benchmarks worth knowing")

    section("The registry")
    for b in BENCHMARKS:
        print(f"  {b.name}")
        print(f"      [{b.category}] {b.measures}")

    section("Per-agent-type grader playbooks (Part 9)")
    for agent_type, playbook in AGENT_PLAYBOOKS.items():
        print(f"  {agent_type:15s}: {playbook}")

    section("Two cautions when reading any leaderboard")
    for c in LEADERBOARD_CAUTIONS:
        print(f"  * {c}")

    print("\n  Lesson: an agentic score measures harness + model + environment, not the model")
    print("  alone. Pick a platform fast, then spend your energy on high-quality tasks and graders.")


if __name__ == "__main__":
    main()
