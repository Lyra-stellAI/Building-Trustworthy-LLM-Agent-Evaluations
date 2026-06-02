"""Part 1 - Building a synthetic evaluation dataset.

The method from the Hugging Face cookbook: sample chunks from your corpus, ask
an LLM to write a factoid question + answer for each, then run *critique agents*
that each score one flaw on a 1-5 scale. Keep only QA pairs scoring >= 4 on all
three (groundedness, relevance, standalone-ness). Expect to discard ~half, so
generate 2x what you need.

Offline, we ship a small knowledge base and a set of *candidate* QA pairs --
some deliberately flawed -- so you can watch the critique filter do its job and
see exactly *why* each rejected pair fails. Pass a real ``LLMClient`` to
:func:`generate_qa` / :func:`critique_qa` to run the genuine cookbook prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .judge import extract_score
from .llm import LLMClient
from .prompts import CRITIQUE_CRITERIA, CRITIQUE_PROMPT, QA_GENERATION_PROMPT

# A tiny in-repo "knowledge base". Each string is one retrievable chunk.
KB_CHUNKS: List[str] = [
    "LLM-as-a-judge works because evaluating text is easier than generating it: "
    "evaluation is closer to focused classification than open-ended generation. "
    "Strong judges reach about 80% agreement with humans on MT-Bench.",
    "A naive judge prompt that asks for a 0-10 float correlates with humans at "
    "only ~0.567. Switching to an anchored 1-4 integer scale with a reasoning "
    "field first lifts correlation to ~0.843.",
    "The Panel of LLM Evaluators (PoLL) replaces one large judge with several "
    "smaller models from disjoint families; it agrees with humans better, shows "
    "less intra-model bias, and runs over 7x cheaper than a GPT-4-class judge.",
    "pass@k is the probability of at least one success in k attempts and rises "
    "with k. pass^k is the probability that all k trials succeed and falls with "
    "k. At a 75% per-trial rate, pass^3 is about 42%.",
    "RAGAS faithfulness decomposes an answer into claims and checks each against "
    "the retrieved context; it is reference-free and the single best catch for "
    "hallucination.",
]


@dataclass
class SyntheticQA:
    """A candidate QA pair plus the flaws it carries (for the offline sim).

    ``flaws`` is a subset of ``{"ungrounded", "irrelevant", "context_dependent"}``.
    In a real pipeline you would not know these; the critique agents *estimate*
    them. Here they let the simulated critique return a faithful score.
    """

    question: str
    answer: str
    context: str
    flaws: Set[str] = field(default_factory=set)
    # Filled in by the critique pass:
    groundedness: Optional[int] = None
    relevance: Optional[int] = None
    standalone: Optional[int] = None

    def passes(self, threshold: int = 4) -> bool:
        scores = (self.groundedness, self.relevance, self.standalone)
        return all(s is not None and s >= threshold for s in scores)


# Hand-authored candidates: ~half are clean, ~half carry exactly one flaw, so the
# filter discards roughly half -- the cookbook's headline ratio.
def candidate_qa_pairs() -> List[SyntheticQA]:
    kb = KB_CHUNKS
    return [
        SyntheticQA(
            "What agreement with humans do strong LLM judges reach on MT-Bench?",
            "About 80%.",
            kb[0],
        ),
        SyntheticQA(
            "Why is evaluating text easier than generating it?",
            "Because evaluation is closer to focused classification than open-ended generation.",
            kb[0],
        ),
        SyntheticQA(
            "What correlation does a naive 0-10 float judge reach with humans?",
            "About 0.567.",
            kb[1],
        ),
        SyntheticQA(
            "What correlation does the improved anchored 1-4 judge reach?",
            "About 0.843.",
            kb[1],
        ),
        SyntheticQA(
            "How much cheaper is a PoLL panel than a GPT-4-class judge?",
            "Over 7x cheaper.",
            kb[2],
        ),
        SyntheticQA(
            "What is pass^3 at a 75% per-trial success rate?",
            "About 42%.",
            kb[3],
        ),
        SyntheticQA(
            "What does RAGAS faithfulness decompose the answer into?",
            "Into claims, each checked against the retrieved context.",
            kb[4],
        ),
        # --- flawed: ungrounded (answer not derivable from the chunk) ---------
        SyntheticQA(
            "Who invented the Elo rating system?",
            "Arpad Elo.",
            kb[0],
            flaws={"ungrounded"},
        ),
        SyntheticQA(
            "What year was the Transformer architecture published?",
            "2017.",
            kb[2],
            flaws={"ungrounded"},
        ),
        # --- flawed: irrelevant (no real user would ask this) ----------------
        SyntheticQA(
            "How many words are in the third sentence of this chunk?",
            "Twelve.",
            kb[1],
            flaws={"irrelevant"},
        ),
        SyntheticQA(
            "What is the second letter of the word 'judge' in the text?",
            "U.",
            kb[2],
            flaws={"irrelevant"},
        ),
        # --- flawed: context-dependent (refers to 'the passage'/'the text') --
        SyntheticQA(
            "According to the passage, what rises with k?",
            "pass@k.",
            kb[3],
            flaws={"context_dependent"},
        ),
        SyntheticQA(
            "What does the text say is the best catch for hallucination?",
            "Faithfulness.",
            kb[4],
            flaws={"context_dependent"},
        ),
        SyntheticQA(
            "In the chunk above, what number is given for pass^3?",
            "About 42%.",
            kb[3],
            flaws={"context_dependent"},
        ),
    ]


# Which flaw each criterion is sensitive to.
_FLAW_FOR_CRITERION = {
    "groundedness": "ungrounded",
    "relevance": "irrelevant",
    "standalone": "context_dependent",
}


def simulate_critique(qa: SyntheticQA, criterion: str) -> int:
    """Deterministic stand-in for a critique LLM pass.

    Returns 5 when the QA pair does not carry the flaw this criterion targets,
    and 2 when it does. A small, stable jitter keeps the scores from being
    suspiciously uniform without ever crossing the >= 4 keep threshold the wrong
    way.
    """
    if criterion not in CRITIQUE_CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")
    flaw = _FLAW_FOR_CRITERION[criterion]
    if flaw in qa.flaws:
        return 2
    # Stable per-(question, criterion) jitter in {4, 5}.
    jitter = (hash((qa.question, criterion)) % 2)  # 0 or 1
    return 4 + jitter


def critique_qa(qa: SyntheticQA, criterion: str, llm: Optional[LLMClient] = None) -> int:
    """Score one criterion for one QA pair (1-5).

    With ``llm=None`` (default) uses the offline simulator. With a real
    ``LLMClient`` it formats the cookbook ``CRITIQUE_PROMPT`` and parses the
    score with the shared :func:`extract_score`.
    """
    if llm is None:
        return simulate_critique(qa, criterion)
    prompt = CRITIQUE_PROMPT.format(criterion=criterion, question=qa.question, context=qa.context)
    raw = llm.complete(prompt)
    score = extract_score(raw, marker="Total rating:")
    if score is None:
        raise ValueError(f"could not parse a critique score from: {raw!r}")
    return int(round(score))


def run_critiques(qa: SyntheticQA, llm: Optional[LLMClient] = None) -> SyntheticQA:
    """Run all three critique passes and record the scores on the QA pair."""
    qa.groundedness = critique_qa(qa, "groundedness", llm)
    qa.relevance = critique_qa(qa, "relevance", llm)
    qa.standalone = critique_qa(qa, "standalone", llm)
    return qa


@dataclass
class FilterReport:
    total: int
    kept: int
    rejected: int
    survivors: List[SyntheticQA]
    rejected_pairs: List[SyntheticQA]

    @property
    def keep_rate(self) -> float:
        return self.kept / self.total if self.total else 0.0


def filter_eval_set(
    candidates: List[SyntheticQA], threshold: int = 4, llm: Optional[LLMClient] = None
) -> FilterReport:
    """Critique every candidate and keep those scoring >= threshold on all three."""
    survivors: List[SyntheticQA] = []
    rejected: List[SyntheticQA] = []
    for qa in candidates:
        run_critiques(qa, llm)
        (survivors if qa.passes(threshold) else rejected).append(qa)
    return FilterReport(
        total=len(candidates),
        kept=len(survivors),
        rejected=len(rejected),
        survivors=survivors,
        rejected_pairs=rejected,
    )


def generate_qa(context: str, llm: LLMClient) -> SyntheticQA:
    """Real-path QA generation using the cookbook prompt (requires an LLM).

    Parses the ``Factoid question:`` / ``Answer:`` block the prompt asks for.
    The offline examples use :func:`candidate_qa_pairs` instead.
    """
    raw = llm.complete(QA_GENERATION_PROMPT.format(context=context))
    question, answer = "", ""
    for line in raw.splitlines():
        low = line.lower()
        if low.startswith("factoid question:"):
            question = line.split(":", 1)[1].strip()
        elif low.startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
    if not question or not answer:
        raise ValueError(f"could not parse QA from generator output: {raw!r}")
    return SyntheticQA(question=question, answer=answer, context=context)


__all__ = [
    "KB_CHUNKS",
    "SyntheticQA",
    "candidate_qa_pairs",
    "simulate_critique",
    "critique_qa",
    "run_critiques",
    "FilterReport",
    "filter_eval_set",
    "generate_qa",
]
