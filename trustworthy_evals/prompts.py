"""Prompt templates, transcribed verbatim from the tutorial.

These are the exact prompts discussed in *Building Trustworthy LLM & Agent
Evaluations*. They are kept in one place so that:

* the simulated demos and the real-LLM adapters use the *same* wording, and
* you can read the canonical prompt next to the code that consumes it.

Nothing in this module calls an LLM; it is pure strings. The
:class:`~trustworthy_evals.llm.SimulatedJudge` ignores the prompt text and works
off latent attributes instead, but the real adapters in
:mod:`trustworthy_evals.llm` format these templates and send them to a model.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Part 1 — Building a synthetic evaluation dataset
# ---------------------------------------------------------------------------

QA_GENERATION_PROMPT = """\
Write a factoid question and its answer given the context below.
The question must be answerable with a specific, concise fact from the context,
phrased the way a user would type it into a search engine.
It MUST NOT reference "the passage" or "the context".
Output:::
Factoid question: (your question)
Answer: (your answer)
Context: {context}
Output:::"""

# One critique pass, parameterized per criterion (groundedness / relevance /
# standalone-ness). Run it once per criterion and keep QA pairs scoring >= 4 on
# all three.
CRITIQUE_PROMPT = """\
Rate the {criterion} of this question on a 1-5 integer scale.
First give your reasoning, then the rating.
Evaluation: (your rationale)
Total rating: (1-5)
Question: {question}
Context: {context}
"""

# The three critique criteria and the plain-language question each one asks.
CRITIQUE_CRITERIA = {
    "groundedness": "Can the question actually be answered from the given context?",
    "relevance": "Would real users care about this question?",
    "standalone": (
        "Is the question intelligible without the surrounding context "
        "(assuming access to documentation)?"
    ),
}


# ---------------------------------------------------------------------------
# Part 2 — Building the judge
# ---------------------------------------------------------------------------

# The tempting one-liner. Continuous 0-10 scale, no rubric, no reasoning field.
# It correlates with humans at ~0.567 in the cookbook -- barely above the
# two-human baseline. Do not ship this.
NAIVE_JUDGE_PROMPT = """\
You will be given a question and a system answer.
Rate how well the answer addresses the question, as a float from 0 to 10.
Feedback:::
Total rating: """

# The improved judge: reasoning-before-score, a small *anchored* integer scale,
# and a parseable marker. Moves correlation to ~0.843 in the cookbook.
IMPROVED_JUDGE_PROMPT = """\
Rate how well the answer addresses the question on a scale of 1 to 4.
1: Terrible - irrelevant or only very partially addresses the question.
2: Mostly unhelpful - misses key aspects.
3: Mostly helpful - addresses the question but could be improved.
4: Excellent - relevant, direct, detailed, addresses all concerns.
Feedback:::
Evaluation: (your reasoning)
Total rating: (1-4)
Question: {question}
Answer: {answer}
"""


# ---------------------------------------------------------------------------
# Part 3 — Pairwise vs. rubric-based scoring
# ---------------------------------------------------------------------------

# Robust pairwise prompt: explicitly tells the judge to ignore length,
# presentation order, and assistant names -- the documented (partial) defenses
# against distracted evaluation and position bias. Run once as (A, B) and once
# as (B, A) and aggregate.
PAIRWISE_PROMPT = """\
Given the instruction and two responses A and B, decide which better satisfies
the criterion below. Ignore length, presentation order, and the assistants' names.
Output exactly one of: [[A]], [[B]], or [[C]] for a tie.
Criterion: {criterion}
Instruction: {instruction}
Response A: {a}
Response B: {b}
"""


# ---------------------------------------------------------------------------
# Part 7 — RAG evaluation end to end
# ---------------------------------------------------------------------------

# Absolute scorer with a Prometheus-style anchored 1-5 rubric, a reference
# answer in the prompt, feedback-before-score, and a [RESULT] marker for robust
# parsing. Note every best practice from Parts 2-3 reappearing here.
RAG_EVAL_PROMPT = """\
###Task Description:
You are given an instruction, a response to evaluate, a reference answer scoring 5,
and a score rubric. Write a feedback assessing the response strictly against the rubric,
then a score from 1 to 5. Format: "Feedback: {{feedback}} [RESULT] {{1-5}}"
###Instruction: {instruction}
###Response to evaluate: {response}
###Reference Answer (Score 5): {reference_answer}
###Score Rubric:
[Is the response correct, accurate, and factual relative to the reference?]
Score 1: Completely incorrect / not factual.
Score 2: Mostly incorrect.
Score 3: Somewhat correct.
Score 4: Mostly correct and factual.
Score 5: Completely correct, accurate, and factual.
###Feedback:"""


__all__ = [
    "QA_GENERATION_PROMPT",
    "CRITIQUE_PROMPT",
    "CRITIQUE_CRITERIA",
    "NAIVE_JUDGE_PROMPT",
    "IMPROVED_JUDGE_PROMPT",
    "PAIRWISE_PROMPT",
    "RAG_EVAL_PROMPT",
]
