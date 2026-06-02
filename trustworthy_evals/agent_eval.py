"""Part 9 - Evaluating agents, not just outputs.

An agent is a system, not a model: it plans, calls tools, keeps state across
turns, and only eventually returns something. That breaks single-output grading
for four reasons -- multi-turn state, tool-call correctness, non-determinism, and
creative out-of-spec solutions.

This module provides a small but complete agent-eval harness:

* the vocabulary as types -- :class:`Task`, :class:`Trial`/:class:`Transcript`,
  graders, :class:`Outcome`;
* the three grader families -- code-based (tests, state, tool-call, numeric,
  regex), model-based (:class:`LLMRubric`), and human (:class:`HumanReview`);
* trajectory evaluation -- :func:`trajectory_match` (strict/unordered/subset/
  superset) and :func:`trajectory_llm_judge`;
* reliability rates -- :func:`run_trials` + ``pass@k`` / ``pass^k``;
* partial credit, the ``fix-auth-bypass-01`` task spec, and demos of the two
  real failure modes (reward hacking via a rigid grader, and a broken task).

The discipline it encodes: prefer deterministic graders, use LLM graders where
necessary, validate with humans, grade the *outcome* over the trajectory, and
report a rate rather than a verdict.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .llm import Response, SimulatedJudge
from .metrics import pass_at_k, pass_at_k_estimate, pass_hat_k, success_rate


# ---------------------------------------------------------------------------
# Vocabulary (Anthropic's terms, matching most platforms)
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, object] = field(default_factory=dict)


@dataclass
class Transcript:
    """The complete record of a trial: outputs, tool calls, and final state."""

    final_output: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_state: Dict[str, object] = field(default_factory=dict)  # the Outcome
    tests_passed: Dict[str, bool] = field(default_factory=dict)
    n_turns: int = 0
    n_tokens: int = 0

    @property
    def tool_sequence(self) -> List[str]:
        return [c.tool for c in self.tool_calls]


@dataclass
class GraderResult:
    name: str
    score: float  # 0-1
    passed: bool
    detail: str = ""


class Grader:
    """A grader scores some aspect of a transcript. ``required`` graders gate a
    hybrid pass; ``weight`` sets the share of partial credit."""

    name = "grader"
    weight = 1.0
    required = False

    def grade(self, t: Transcript) -> GraderResult:  # pragma: no cover - interface
        raise NotImplementedError


# ---- code-based graders (the backbone: fast, cheap, objective) -------------


@dataclass
class OutputContains(Grader):
    substring: str
    weight: float = 1.0
    required: bool = False
    name: str = "output_contains"

    def grade(self, t: Transcript) -> GraderResult:
        ok = self.substring in t.final_output
        return GraderResult(self.name, float(ok), ok, f"looking for {self.substring!r}")


@dataclass
class NumericMatch(Grader):
    """Match a number with a tolerance. The cure for the rigid-grader failure
    mode: rejecting ``96.12`` when the key was ``96.124991...``"""

    expected: float
    tol: float = 1e-2
    weight: float = 1.0
    required: bool = False
    name: str = "numeric_match"

    def grade(self, t: Transcript) -> GraderResult:
        nums = re.findall(r"-?\d+(?:\.\d+)?", t.final_output)
        if not nums:
            return GraderResult(self.name, 0.0, False, "no number found")
        value = float(nums[0])
        ok = abs(value - self.expected) <= self.tol
        return GraderResult(self.name, float(ok), ok, f"|{value}-{self.expected}|<= {self.tol}")


@dataclass
class TestsPass(Grader):
    """Binary fail-to-pass / pass-to-pass tests -- the coding-agent backbone."""

    __test__ = False  # not a pytest test class despite the "Test" prefix

    required_tests: List[str] = field(default_factory=list)
    weight: float = 1.0
    required: bool = False
    name: str = "tests_pass"

    def grade(self, t: Transcript) -> GraderResult:
        if not self.required_tests:
            return GraderResult(self.name, 1.0, True, "no tests required")
        passed = sum(1 for name in self.required_tests if t.tests_passed.get(name, False))
        score = passed / len(self.required_tests)
        return GraderResult(self.name, score, score == 1.0, f"{passed}/{len(self.required_tests)} tests pass")


@dataclass
class StateCheck(Grader):
    """Did the environment actually change? Verify the outcome, not the claim."""

    expected: Dict[str, object] = field(default_factory=dict)
    weight: float = 1.0
    required: bool = False
    name: str = "state_check"

    def grade(self, t: Transcript) -> GraderResult:
        ok = all(t.final_state.get(k) == v for k, v in self.expected.items())
        return GraderResult(self.name, float(ok), ok, f"expected state {self.expected}")


@dataclass
class ToolCallsInclude(Grader):
    """Light guardrail: required tools were used (NOT a rigid script)."""

    required_tools: List[str] = field(default_factory=list)
    weight: float = 1.0
    required: bool = False
    name: str = "tool_calls_include"

    def grade(self, t: Transcript) -> GraderResult:
        seq = set(t.tool_sequence)
        present = sum(1 for tool in self.required_tools if tool in seq)
        score = present / len(self.required_tools) if self.required_tools else 1.0
        return GraderResult(self.name, score, score == 1.0, f"{present}/{len(self.required_tools)} tools used")


@dataclass
class ForbiddenTool(Grader):
    """A safety check: a forbidden tool was never called."""

    tool: str = ""
    weight: float = 1.0
    required: bool = True
    name: str = "forbidden_tool"

    def grade(self, t: Transcript) -> GraderResult:
        ok = self.tool not in t.tool_sequence
        return GraderResult(self.name, float(ok), ok, f"{self.tool!r} must not be called")


# ---- model-based grader (LLM-as-judge) ------------------------------------


@dataclass
class LLMRubric(Grader):
    """Rubric scoring by an LLM judge -- for nuance code graders can't reach
    (code quality, tone, groundedness). Non-deterministic; calibrate to humans.

    Offline, scores ``transcript.final_state['quality']`` (a stand-in for what a
    real judge would read off the transcript) through the simulated judge.
    """

    criterion: str = "code quality"
    weight: float = 1.0
    required: bool = False
    judge: SimulatedJudge = field(default_factory=lambda: SimulatedJudge(noise=0.03))
    name: str = "llm_rubric"

    def grade(self, t: Transcript) -> GraderResult:
        quality = float(t.final_state.get("quality", 0.5))  # type: ignore[arg-type]
        score_1_5 = self.judge.score_absolute(Response(text=t.final_output, quality=quality), scale=(1, 5))
        score = (score_1_5 - 1) / 4
        return GraderResult(self.name, score, score >= 0.5, f"{self.criterion}: {score_1_5}/5")


# ---- human grader (gold standard; used to calibrate) ----------------------


@dataclass
class HumanReview(Grader):
    """A stand-in for SME review: a fixed verdict you supply (e.g. from a label)."""

    verdict: bool = True
    weight: float = 1.0
    required: bool = False
    name: str = "human_review"

    def grade(self, t: Transcript) -> GraderResult:
        return GraderResult(self.name, float(self.verdict), self.verdict, "human SME verdict")


# ---------------------------------------------------------------------------
# Task + scoring (weighted / binary / hybrid, with partial credit)
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    description: str
    graders: List[Grader]
    mode: str = "hybrid"  # 'binary' | 'weighted' | 'hybrid'
    pass_threshold: float = 0.6


@dataclass
class TaskResult:
    task_id: str
    score: float  # 0-1 weighted (partial credit)
    passed: bool
    grader_results: List[GraderResult]

    def detail(self) -> str:
        return "; ".join(f"{g.name}={g.score:.2f}{'*' if not g.passed else ''}" for g in self.grader_results)


def grade_task(task: Task, t: Transcript) -> TaskResult:
    """Score a transcript against a task.

    * ``binary``   -- every grader must pass.
    * ``weighted`` -- weighted average of grader scores (partial credit).
    * ``hybrid``   -- all ``required`` graders must pass *and* the weighted score
      of the rest must clear ``pass_threshold``. This is what gives "identified
      the problem and verified the customer but failed the refund" more credit
      than "failed immediately".
    """
    results = [g.grade(t) for g in task.graders]
    by_grader = list(zip(task.graders, results))

    weighted_num = sum(g.weight * r.score for g, r in by_grader)
    weighted_den = sum(g.weight for g, r in by_grader) or 1.0
    weighted = weighted_num / weighted_den

    if task.mode == "binary":
        passed = all(r.passed for r in results)
    elif task.mode == "weighted":
        passed = weighted >= task.pass_threshold
    elif task.mode == "hybrid":
        required_ok = all(r.passed for g, r in by_grader if g.required)
        passed = required_ok and weighted >= task.pass_threshold
    else:
        raise ValueError(f"unknown task mode {task.mode!r}")

    return TaskResult(task.id, weighted, passed, results)


# ---------------------------------------------------------------------------
# Trajectory evaluation (grade the path only where you truly care)
# ---------------------------------------------------------------------------


def trajectory_match(actual: Sequence[str], reference: Sequence[str], mode: str = "strict") -> bool:
    """Compare a tool-call sequence against a reference (deterministic, no LLM).

    * ``strict``    -- identical order and set.
    * ``unordered`` -- same multiset of tools, any order.
    * ``subset``    -- every actual tool appears in the reference (no extras).
    * ``superset``  -- every reference tool appears in the actual (extras allowed).
    """
    a, r = list(actual), list(reference)
    if mode == "strict":
        return a == r
    if mode == "unordered":
        return sorted(a) == sorted(r)
    if mode == "subset":
        return set(a).issubset(set(r))
    if mode == "superset":
        return set(r).issubset(set(a))
    raise ValueError(f"unknown trajectory match mode {mode!r}")


def trajectory_llm_judge(
    t: Transcript, judge: Optional[SimulatedJudge] = None, efficiency_hint: Optional[float] = None
) -> float:
    """An LLM reviews the execution path for efficiency/appropriateness (0-1).

    Flexible and needs no reference, but non-deterministic and costs a call.
    Offline, scores a supplied efficiency signal (e.g. fewer redundant tool calls
    is better) through the judge.
    """
    judge = judge or SimulatedJudge(noise=0.03)
    quality = efficiency_hint if efficiency_hint is not None else float(t.final_state.get("path_quality", 0.5))  # type: ignore[arg-type]
    return (judge.score_absolute(Response(text="trajectory", quality=quality), scale=(1, 5)) - 1) / 4


# ---------------------------------------------------------------------------
# Non-determinism: run multiple trials, report a rate
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    index: int
    transcript: Transcript
    result: TaskResult


@dataclass
class ReliabilityReport:
    task_id: str
    k: int
    observed_rate: float
    pass_at_k: float
    pass_hat_k: float
    pass_at_k_unbiased: float
    trials: List[Trial]

    def summary(self) -> str:
        return (
            f"{self.task_id}: per-trial={self.observed_rate:.2f}  "
            f"pass@{self.k}={self.pass_at_k:.2f}  pass^{self.k}={self.pass_hat_k:.2f}"
        )


Agent = Callable[[Task, random.Random], Transcript]


def run_trials(task: Task, agent: Agent, k: int = 10, seed: int = 0) -> ReliabilityReport:
    """Run k isolated trials and report both reliability rates.

    Each trial gets its own RNG-seeded run; in a real harness each would also get
    an isolated environment, because leftover state and resource exhaustion cause
    correlated, fake failures.
    """
    trials: List[Trial] = []
    for i in range(k):
        rng = random.Random(f"{seed}-{i}")
        transcript = agent(task, rng)
        result = grade_task(task, transcript)
        trials.append(Trial(i, transcript, result))
    outcomes = [tr.result.passed for tr in trials]
    p = success_rate(outcomes)
    c = sum(outcomes)
    return ReliabilityReport(
        task_id=task.id,
        k=k,
        observed_rate=p,
        pass_at_k=pass_at_k(p, k),
        pass_hat_k=pass_hat_k(p, k),
        pass_at_k_unbiased=pass_at_k_estimate(k, c, k) if c else 0.0,
        trials=trials,
    )


def make_flaky_agent(p_success: float, quality_on_success: float = 0.85) -> Agent:
    """A toy agent that completes the auth-bypass task with probability p."""

    def agent(task: Task, rng: random.Random) -> Transcript:
        success = rng.random() < p_success
        return Transcript(
            final_output="Rejected empty/null passwords; added auth_blocked log event." if success else "Patched.",
            tool_calls=[ToolCall("edit_file", {"path": "auth.py"}), ToolCall("run_tests")],
            final_state={
                "security_logs": {"event_type": "auth_blocked"} if success else {},
                "quality": quality_on_success if success else 0.5,
            },
            tests_passed={"test_empty_pw_rejected.py": success, "test_null_pw_rejected.py": success},
            n_turns=rng.randint(4, 9),
            n_tokens=rng.randint(2000, 6000),
        )

    return agent


# ---------------------------------------------------------------------------
# The fix-auth-bypass-01 task spec (Part 9, made concrete)
# ---------------------------------------------------------------------------

AUTH_BYPASS_SPEC_YAML = """\
task:
  id: fix-auth-bypass-01
  desc: "Reject empty/null passwords in the auth path; add a security log event."
  graders:
    - type: deterministic_tests       # the backbone: outcome correctness
      required: [test_empty_pw_rejected.py, test_null_pw_rejected.py]
    - type: llm_rubric                 # code quality, calibrated to humans
      rubric: prompts/code_quality.md
    - type: static_analysis
      commands: [ruff, mypy, bandit]
    - type: state_check                # did the environment actually change?
      expect: {security_logs: {event_type: "auth_blocked"}}
    - type: tool_calls                 # light guardrails, NOT a rigid script
      required: [{tool: edit_file}, {tool: run_tests}]
  tracked_metrics:
    transcript: [n_turns, n_tool_calls, n_total_tokens]
    latency:    [time_to_first_token, output_tokens_per_sec]
"""


def auth_bypass_task() -> Task:
    """The YAML spec above, built as a runnable :class:`Task` (hybrid scoring)."""
    return Task(
        id="fix-auth-bypass-01",
        description="Reject empty/null passwords in the auth path; add a security log event.",
        graders=[
            TestsPass(["test_empty_pw_rejected.py", "test_null_pw_rejected.py"], weight=2.0, required=True),
            LLMRubric(criterion="code quality", weight=1.0),
            StateCheck({"security_logs": {"event_type": "auth_blocked"}}, weight=1.0, required=True),
            ToolCallsInclude(["edit_file", "run_tests"], weight=0.5),
        ],
        mode="hybrid",
        pass_threshold=0.6,
    )


# ---------------------------------------------------------------------------
# Capability vs. regression evals
# ---------------------------------------------------------------------------


def classify_eval_health(kind: str, pass_rate: float) -> str:
    """Capability evals should start low (a hill to climb); regression evals sit
    near 100% (a drop signals a break)."""
    if kind == "capability":
        if pass_rate >= 0.9:
            return "saturated - graduate into the regression suite"
        if pass_rate <= 0.4:
            return "healthy - plenty of headroom to climb"
        return "in progress"
    if kind == "regression":
        return "healthy" if pass_rate >= 0.95 else "REGRESSION - something broke"
    raise ValueError("kind must be 'capability' or 'regression'")


def likely_broken_task(pass_at_100: float) -> bool:
    """With frontier models, a 0% pass@100 usually means a broken task or grader,
    not an incapable agent. Read the transcripts before trusting the score."""
    return pass_at_100 == 0.0


# ---- per-agent-type grader playbooks (Part 9) -----------------------------

AGENT_PLAYBOOKS: Dict[str, str] = {
    "coding": "binary fail-to-pass/pass-to-pass tests + code-quality LLM rubric + static analysis",
    "conversational": "verifiable end-state + transcript constraint (<N turns) + tone rubric; simulate the user",
    "research": "groundedness + coverage + source-quality + exact-match sub-questions + synthesis rubric",
    "computer_use": "outcome incl. backend state (order actually placed, not just confirmation page rendered)",
}


__all__ = [
    "ToolCall",
    "Transcript",
    "GraderResult",
    "Grader",
    "OutputContains",
    "NumericMatch",
    "TestsPass",
    "StateCheck",
    "ToolCallsInclude",
    "ForbiddenTool",
    "LLMRubric",
    "HumanReview",
    "Task",
    "TaskResult",
    "grade_task",
    "trajectory_match",
    "trajectory_llm_judge",
    "Trial",
    "ReliabilityReport",
    "run_trials",
    "make_flaky_agent",
    "AUTH_BYPASS_SPEC_YAML",
    "auth_bypass_task",
    "classify_eval_health",
    "likely_broken_task",
    "AGENT_PLAYBOOKS",
]
