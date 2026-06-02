"""Part 7 - Putting it together: RAG evaluation end to end.

A RAG pipeline has many knobs -- chunk size, embeddings, reranking, reader model
-- and tuning any of them is pointless if you can't measure the effect. This
module is a small but *real* RAG system (a token-overlap retriever over a real
corpus, a reader whose answer quality depends on whether the answer-bearing
chunk was actually retrieved) scored by an absolute, anchored judge using the
``RAG_EVAL_PROMPT`` from Part 7.

:func:`sweep_configs` sweeps chunk size / rerank / reader and scores each, so you
can watch the empirical conclusion emerge: there is no single good recipe, but
chunk-size tuning tends to be cheap and high-impact.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .llm import LLMClient, Response, SimulatedJudge
from .judge import extract_score
from .metrics import normalize
from .prompts import RAG_EVAL_PROMPT

# A small corpus about evaluation. Each document is several sentences; questions
# below target a specific answer-bearing sentence in one document.
DOCUMENTS: List[str] = [
    (
        "Inter-rater agreement sets a ceiling on judge quality. When two human "
        "annotators label the same examples, their correlation bounds how well any "
        "automated judge can match the ground truth. In one cookbook run the "
        "annotators correlated at only zero point five six three. Low agreement "
        "usually means the rating criteria are not crisp enough. You can reduce the "
        "noise by averaging the scores or keeping only examples where annotators agree."
    ),
    (
        "A panel of evaluators replaces a single judge with several smaller models. "
        "Drawing the panel from disjoint families reduces intra-model bias because no "
        "one family dominates the verdict. The panel agrees with humans better and "
        "runs over seven times cheaper than a single large judge. Aggregate discrete "
        "verdicts by majority vote and graded scores by average pooling."
    ),
    (
        "Pairwise comparison shows two outputs and picks the better one, so its "
        "signature failure is position bias. The standard defense runs each pair "
        "twice with the order swapped and aggregates the two calls. Absolute scoring "
        "instead rates one output against an anchored rubric and keeps penalizing "
        "violations even when the prose is slick. For verifiable criteria, prefer "
        "absolute scoring."
    ),
    (
        "Agent reliability is reported as a rate, not a single verdict. The metric "
        "pass at k is the probability of at least one success in k attempts and rises "
        "with k. The metric pass hat k is the probability that all k trials succeed "
        "and falls with k. At a seventy five percent per-trial rate, pass hat three "
        "is about forty two percent. Reliability-critical deployments watch pass hat k."
    ),
]


@dataclass
class EvalItem:
    question: str
    answer_marker: str  # a phrase that appears in the answer-bearing sentence
    reference_answer: str
    doc_index: int


EVAL_SET: List[EvalItem] = [
    EvalItem("What did the two annotators correlate at in the cookbook run?",
             "zero point five six three", "About 0.563.", 0),
    EvalItem("How do you reduce noise in human labels?",
             "averaging the scores or keeping only examples", "Average the scores or keep agreement-only examples.", 0),
    EvalItem("How much cheaper is the panel than a single large judge?",
             "seven times cheaper", "Over seven times cheaper.", 1),
    EvalItem("How should graded scores be aggregated across a panel?",
             "average pooling", "By average pooling.", 1),
    EvalItem("What is the signature failure mode of pairwise comparison?",
             "position bias", "Position bias.", 2),
    EvalItem("Which scoring protocol is preferred for verifiable criteria?",
             "prefer absolute scoring", "Absolute scoring.", 2),
    EvalItem("What is pass hat three at a seventy five percent per-trial rate?",
             "about forty two percent", "About 42%.", 3),
    EvalItem("Which metric rises with k?",
             "at least one success in k attempts", "pass at k.", 3),
]

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "is", "are", "was", "were", "and", "or",
    "you", "your", "how", "what", "which", "do", "does", "did", "with", "for", "by", "it",
    "its", "as", "that", "this", "than", "only", "so", "when", "their", "they", "one", "two",
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def chunk_document(text: str, size_words: int) -> List[str]:
    """Split a document into chunks of about ``size_words`` words."""
    words = text.split()
    return [" ".join(words[i : i + size_words]) for i in range(0, len(words), size_words)] or [text]


@dataclass
class RagConfig:
    chunk_size: int = 40
    rerank: bool = False
    reader: str = "strong"  # 'strong' or 'weak'
    embeddings: str = "gte-small"
    top_k: int = 2

    def label(self) -> str:
        return f"chunk={self.chunk_size},rerank={self.rerank},reader={self.reader},emb={self.embeddings}"


class KeywordRetriever:
    """A real (if simple) retriever: rank chunks by query-token overlap."""

    def __init__(self, chunks: Sequence[str]):
        self.chunks = list(chunks)
        self._tokens = [set(tokenize(c)) for c in self.chunks]

    def _score(self, q_tokens: Sequence[str], idx: int) -> int:
        return sum(1 for t in q_tokens if t in self._tokens[idx])

    def retrieve(self, query: str, k: int, rerank: bool = False) -> List[str]:
        q = tokenize(query)
        scored = sorted(range(len(self.chunks)), key=lambda i: self._score(q, i), reverse=True)
        if rerank:
            # Reranking widens the candidate pool, then re-scores it with a
            # finer signal (token overlap weighted by chunk brevity -> precision).
            pool = scored[: max(k * 3, k)]
            pool.sort(key=lambda i: self._score(q, i) / (1 + len(self._tokens[i]) / 50.0), reverse=True)
            return [self.chunks[i] for i in pool[:k]]
        return [self.chunks[i] for i in scored[:k]]


def run_rag(config: RagConfig, seed: int = 0) -> List[Tuple[EvalItem, Response, bool]]:
    """Run the pipeline over the eval set, returning (item, answer, retrieved_hit).

    The reader's answer quality depends on whether the answer-bearing chunk was
    actually retrieved -- a *real* check against the retriever's output -- plus
    the reader model's skill. A miss caps quality; a hit lets a strong reader
    shine.
    """
    chunks = [c for doc in DOCUMENTS for c in chunk_document(doc, config.chunk_size)]
    retriever = KeywordRetriever(chunks)
    reader_skill = 0.9 if config.reader == "strong" else 0.6
    rng = random.Random(seed)

    out: List[Tuple[EvalItem, Response, bool]] = []
    for item in EVAL_SET:
        retrieved = retriever.retrieve(item.question, config.top_k, rerank=config.rerank)
        hit = any(item.answer_marker in c for c in retrieved)
        base = reader_skill if hit else 0.40  # can't answer well without the evidence
        quality = max(0.02, min(0.99, base + rng.gauss(0, 0.03)))
        answer = Response(text=f"[answer to: {item.question}]", quality=quality, source_model=config.reader)
        out.append((item, answer, hit))
    return out


def score_with_rag_judge(
    runs: Sequence[Tuple[EvalItem, Response, bool]],
    judge: Optional[SimulatedJudge] = None,
    llm: Optional[LLMClient] = None,
) -> float:
    """Score answers with the anchored 1-5 RAG judge, normalized to 0-1.

    Offline (``llm=None``) the simulated judge scores answer quality. With a real
    ``LLMClient`` it formats ``RAG_EVAL_PROMPT`` (reference answer, anchored
    rubric, ``[RESULT]`` marker) and parses with :func:`extract_score`.
    """
    judge = judge or SimulatedJudge(family="judge", noise=0.03)  # different family than reader
    total = 0.0
    for item, answer, _hit in runs:
        if llm is None:
            score = judge.score_absolute(answer, scale=(1, 5))
        else:
            prompt = RAG_EVAL_PROMPT.format(
                instruction=item.question,
                response=answer.text,
                reference_answer=item.reference_answer,
            )
            parsed = extract_score(llm.complete(prompt), marker="[RESULT]")
            if parsed is None:
                raise ValueError("RAG judge produced no parseable [RESULT] score")
            score = parsed
        total += normalize(score, 1, 5)
    return total / len(runs)


@dataclass
class SweepResult:
    config: RagConfig
    score: float
    hit_rate: float


def sweep_configs(configs: Optional[Sequence[RagConfig]] = None, seed: int = 0) -> List[SweepResult]:
    """Sweep configurations and score each end to end (sorted best-first)."""
    if configs is None:
        configs = [
            RagConfig(chunk_size=cs, rerank=rr, reader=rd)
            for cs in (12, 40)
            for rr in (False, True)
            for rd in ("strong", "weak")
        ]
    judge = SimulatedJudge(family="judge", noise=0.03)
    results: List[SweepResult] = []
    for cfg in configs:
        runs = run_rag(cfg, seed=seed)
        score = score_with_rag_judge(runs, judge=judge)
        hit_rate = sum(1 for _, _, hit in runs if hit) / len(runs)
        results.append(SweepResult(cfg, score, hit_rate))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


__all__ = [
    "DOCUMENTS",
    "EvalItem",
    "EVAL_SET",
    "tokenize",
    "chunk_document",
    "RagConfig",
    "KeywordRetriever",
    "run_rag",
    "score_with_rag_judge",
    "SweepResult",
    "sweep_configs",
]
