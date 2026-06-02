"""Part 11 - Agent benchmarks worth knowing.

Public benchmarks won't tell you whether *your* agent works -- build your own
task bank from real failures for that -- but they are the shared language of the
field. This module is a small registry plus the two cautions every leaderboard
reader needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Benchmark:
    name: str
    category: str
    measures: str


BENCHMARKS: List[Benchmark] = [
    Benchmark("SWE-bench / SWE-bench Verified", "Coding",
              "Resolve real GitHub issues; graded by running the repo's test suite"),
    Benchmark("Terminal-Bench (2.0)", "Coding / systems",
              "End-to-end terminal tasks (build software, train models) in a containerized sandbox"),
    Benchmark("tau-bench / tau2-bench", "Conversational + tools",
              "Multi-turn tool-agent-user interaction with a simulated user and domain-policy adherence"),
    Benchmark("BFCL (Berkeley Function-Calling Leaderboard)", "Function calling / tool use",
              "Accuracy of function calls -- simple, parallel, and multi-turn; stateful reasoning"),
    Benchmark("ToolBench / ToolLLM", "Tool use",
              "Tool use across 16k+ real RESTful APIs; multi-step invocation and the ability to abstain"),
    Benchmark("GAIA", "General assistant",
              "Compound, multi-step reasoning + tool use; questions easy to verify but hard to solve"),
    Benchmark("AgentBench", "General / diagnostic",
              "LLM-as-agent reasoning across multiple environments; breadth-first architecture diagnostic"),
    Benchmark("WebArena / VisualWebArena", "Web / computer use",
              "Autonomous navigation of realistic sites, graded on URL/page state and backend changes"),
    Benchmark("OSWorld / AndroidWorld", "Computer use",
              "Full OS / mobile control, graded by inspecting files, configs, and databases after the task"),
    Benchmark("BrowseComp", "Research / web",
              "Finding hard-to-locate, easy-to-verify information across the open web"),
]

# The two cautions when reading any leaderboard.
LEADERBOARD_CAUTIONS = [
    "Saturation: once scores cross ~80%, small increments can hide large real "
    "gains, and contamination/over-fitting becomes a risk.",
    "Harness & infrastructure sensitivity: an agentic score measures harness + "
    "model + environment, not the model alone -- compare like-for-like.",
]


def by_category(category: str) -> List[Benchmark]:
    return [b for b in BENCHMARKS if category.lower() in b.category.lower()]


__all__ = ["Benchmark", "BENCHMARKS", "LEADERBOARD_CAUTIONS", "by_category"]
