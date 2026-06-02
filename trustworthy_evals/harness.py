"""Part 12 - Open-source agent frameworks: LangChain, LangGraph, Deep Agents.

The reframing this part turns into code: ``agent = model + harness``. The model
is the frozen weights; the harness is everything wrapped around them -- system
prompt, tools and how their results are fed back, skills, context management,
sub-agent delegation, retry/verification, execution flow. The practical upshot
is that **most of an agent's behavior, and most of your headroom to improve it,
lives in the harness, not the weights.**

This module makes that measurable:

* :func:`harness_engineering_demo` reproduces LangChain's result -- changing only
  the harness (model fixed) moved ``deepagents-cli`` +13.7 points on
  Terminal-Bench 2.0 (52.8 -> 66.5), roughly rank 30 into the top 5.
* :func:`skills_ablation_demo` reproduces the curated-skills finding: ~82% task
  completion with a skill set vs. ~9% without.
* :data:`DEEP_AGENT_COMPONENTS` / :data:`DEEP_AGENT_MIDDLEWARE` document the
  batteries-included Deep Agents harness as data.
* :func:`boundary_enforcement` encodes the "trust the LLM, enforce at the tool /
  sandbox level" stance -- the same lesson as Part 9's bypass-resistant graders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# The three layers of the open-source stack (kept distinct, as the guide urges)
# ---------------------------------------------------------------------------

STACK_LAYERS: Dict[str, str] = {
    "LangChain": "the building blocks - standard interfaces to chat models, tools, "
    "retrievers, plus create_agent for a thin agent loop (provider-agnostic).",
    "LangGraph": "the runtime - models an agent as a stateful graph with durable "
    "execution, streaming, persistence/checkpointing, human-in-the-loop. Solves "
    "Part 9's multi-turn-state problem.",
    "Deep Agents": "the batteries-included harness on top of LangGraph "
    "(create_deep_agent): planning, virtual filesystem, sub-agents, middleware.",
}

# What Deep Agents provides out of the box.
DEEP_AGENT_COMPONENTS: Dict[str, str] = {
    "planning": "write_todos tool decomposes a task into a tracked list before execution.",
    "virtual_filesystem": "ls/read_file/write_file/edit_file/glob/grep so large "
    "intermediate results are offloaded to files instead of bloating the context window.",
    "sub_agents": "a task tool spawns general or specialized agents in isolated context windows.",
    "middleware": "a LangGraph state machine wrapped in middleware that can add tools, "
    "wrap model calls (inject prompts), and wrap tool calls (post-process results).",
}

# Pluggable filesystem backends (ephemeral -> durable -> sandboxed).
FILESYSTEM_BACKENDS: List[str] = [
    "ephemeral in-graph state",
    "local disk",
    "LangGraph store (cross-thread memory)",
    "LangSmith Context Hub repo (persistence)",
    "isolated sandbox (Modal/Daytona/Deno) - shell-capable backends add an execute tool",
]

# Default middleware stack, applied in order.
DEEP_AGENT_MIDDLEWARE: List[str] = [
    "filesystem",
    "sub-agent",
    "todo",
    "summarization",
    "human-in-the-loop",
]


# ---------------------------------------------------------------------------
# agent = model + harness, made measurable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """One editable harness surface and the points it contributes when enabled.

    A "surface" is a real thing the agent loads at runtime: the system prompt, a
    tool's description, a skills file, a piece of middleware. Each is a knob.
    """

    name: str
    kind: str  # 'prompt' | 'tools' | 'skills' | 'context' | 'middleware'
    points: float  # contribution to the metric when quality == 1.0


@dataclass
class Harness:
    """A model held fixed plus a set of harness surfaces (the knobs you turn).

    ``base`` is the score the frozen model gets on a thin harness; each surface
    adds its points scaled by quality. Score is clamped to ``scale``.
    """

    base: float
    surfaces: Dict[str, Surface] = field(default_factory=dict)
    quality: Dict[str, float] = field(default_factory=dict)  # surface name -> 0..1
    scale: float = 100.0

    def with_surface(self, surface: Surface, quality: float = 1.0) -> "Harness":
        surfaces = dict(self.surfaces, **{surface.name: surface})
        q = dict(self.quality, **{surface.name: quality})
        return Harness(self.base, surfaces, q, self.scale)

    def score(self) -> float:
        total = self.base + sum(
            s.points * self.quality.get(name, 0.0) for name, s in self.surfaces.items()
        )
        return max(0.0, min(self.scale, total))


# The four harness-only changes LangChain reported on Terminal-Bench 2.0.
TERMINAL_BENCH_SURFACES: List[Surface] = [
    Surface("build_verify_loop", "middleware", 4.7),
    Surface("directory_map_context", "context", 3.0),
    Surface("loop_detection_middleware", "middleware", 2.5),
    Surface("reasoning_sandwich", "prompt", 3.5),
]


@dataclass
class HarnessGain:
    baseline_score: float
    improved_score: float

    @property
    def delta(self) -> float:
        return self.improved_score - self.baseline_score


def harness_engineering_demo() -> HarnessGain:
    """Model held fixed (gpt-5.2-codex); only the harness changes.

    Reproduces LangChain's +13.7-point Terminal-Bench 2.0 result (52.8 -> 66.5).
    The point is structural: the same weights, wrapped in a better harness, jump
    from roughly rank 30 into the top 5.
    """
    baseline = Harness(base=52.8)  # thin harness (create_agent style)
    improved = baseline
    for surface in TERMINAL_BENCH_SURFACES:
        improved = improved.with_surface(surface, quality=1.0)
    return HarnessGain(baseline.score(), improved.score())


def skills_ablation_demo() -> HarnessGain:
    """Curated skills are a harness surface with outsized impact.

    Reproduces LangChain's Deep Agents skills finding: ~82% task completion with
    a curated skill set vs. ~9% without -- model unchanged.
    """
    no_skills = Harness(base=9.0)
    curated = no_skills.with_surface(Surface("curated_skills", "skills", 73.0), quality=1.0)
    return HarnessGain(no_skills.score(), curated.score())


# ---------------------------------------------------------------------------
# "Trust the LLM, enforce at the tool / sandbox level" (ties to Part 9)
# ---------------------------------------------------------------------------


def boundary_enforcement(*, sandbox_enforced: bool, model_complies: bool) -> bool:
    """Does a boundary actually hold?

    Deep Agents follows a "trust the LLM" model and expects you to enforce
    boundaries at the tool and sandbox level, not by asking the model to police
    itself. A sandbox-enforced limit holds regardless of the model; a
    prompt-policed limit holds only if the model chooses to comply. This is the
    same lesson as Part 9's "make graders resistant to bypasses": safety and
    correctness are properties of the harness and environment, not promises from
    the model.
    """
    return True if sandbox_enforced else bool(model_complies)


__all__ = [
    "STACK_LAYERS",
    "DEEP_AGENT_COMPONENTS",
    "FILESYSTEM_BACKENDS",
    "DEEP_AGENT_MIDDLEWARE",
    "Surface",
    "Harness",
    "TERMINAL_BENCH_SURFACES",
    "HarnessGain",
    "harness_engineering_demo",
    "skills_ablation_demo",
    "boundary_enforcement",
]
