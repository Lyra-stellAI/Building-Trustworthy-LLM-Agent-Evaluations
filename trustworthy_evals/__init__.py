"""trustworthy_evals -- runnable companion code for *Building Trustworthy LLM &
Agent Evaluations*.

Each module maps to one part of the guide:

============  ===================================================================
Module        Tutorial part
============  ===================================================================
prompts       The canonical prompt templates (verbatim)
datasets      Part 1 - Building a synthetic evaluation dataset
judge         Part 2 - Building the judge (and validating it first)
protocols     Part 3 - Pairwise vs. rubric-based scoring
biases        Part 4 - Known biases and how to mitigate them
jury          Part 5 - Panels and juries
calibration   Part 6 - Calibration into a trustworthy instrument
rag_eval      Part 7 - RAG evaluation end to end
frameworks    Part 8 - RAGAS- and DeepEval-style metrics
agent_eval    Part 9 - Evaluating agents, not just outputs
benchmarks    Part 11 - Agent benchmarks worth knowing
metrics       Shared statistics (correlation, kappa, pass@k / pass^k)
llm           The simulated judge + real-LLM adapters
============  ===================================================================

Everything runs offline against a deterministic :class:`~trustworthy_evals.llm.SimulatedJudge`;
swap in :class:`~trustworthy_evals.llm.AnthropicJudge` / ``OpenAIJudge`` for a
real evaluation.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    agent_eval,
    benchmarks,
    biases,
    calibration,
    datasets,
    frameworks,
    judge,
    jury,
    llm,
    metrics,
    prompts,
    protocols,
    rag_eval,
)
from .llm import Response, SimulatedJudge

__all__ = [
    "__version__",
    "Response",
    "SimulatedJudge",
    "prompts",
    "datasets",
    "judge",
    "protocols",
    "biases",
    "jury",
    "calibration",
    "rag_eval",
    "frameworks",
    "agent_eval",
    "benchmarks",
    "metrics",
    "llm",
]
