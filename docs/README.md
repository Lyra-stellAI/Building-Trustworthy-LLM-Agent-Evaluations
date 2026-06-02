# Building Trustworthy LLM & Agent Evaluations — the source guide

This folder holds the practitioner's guide this repository is built from, plus
the concept-to-code reference.

📄 **[Building-Trustworthy-LLM-and-Agent-Evaluations.pdf](./Building-Trustworthy-LLM-and-Agent-Evaluations.pdf)** · 31 pages · compiled June 2026

> A practitioner's guide to LLM-as-a-judge and agent evaluation: building,
> validating, and operationalizing evaluators, plus the methodologies and
> frameworks the major labs actually use — LangChain/LangSmith,
> Microsoft/AutoGen, Anthropic/Claude Code, and OpenAI. It synthesizes the
> Hugging Face cookbooks, LangChain's calibration workflow, the COLM 2025
> protocol-bias paper (Tripathi et al.), the PoLL jury paper (Verga et al.),
> Anthropic's agent-eval methodology, and the RAGAS and DeepEval frameworks. It
> closes with the open-source agent stack — LangChain, LangGraph, and Deep
> Agents — and two worked examples (an LLM-wiki Deep Agent and the
> better-harness eval-driven optimizer) that ground the concepts in real code.

The central thesis: LLM-as-a-judge scales evaluation past surface metrics
(ROUGE/BLEU), but **it does not work out of the box**. A naive judge is
unreliable, the framing introduces systematic bias, and prompt-tweaking alone
won't get you to an evaluator you'd trust for a shipping decision. A useful
framing throughout — *a judge is a scoring mechanism, and like any mechanism it
can be gamed.* The guide walks the full path from a synthetic dataset to
calibrated judges, juries, RAG and agent evaluation, and harness optimization.

## Contents

| # | Section | In one line |
| --- | --- | --- |
| — | The problem evaluation is actually solving | Why surface metrics fail and LLM-as-judge scales (≈80% human agreement, but not out of the box). |
| — | The two halves of an evaluation pipeline | Every setup needs a dataset and an evaluator; LLMs can help build both. |
| 1 | Building a synthetic evaluation dataset | Generate QA from your corpus; filter with critique agents (groundedness / relevance / standalone-ness, keep ≥4/5). Expect to discard ~half. |
| 2 | Building the judge (and why you validate it first) | Validate against humans before trusting; reasoning-first + anchored integer scale moves a naive judge **0.567 → 0.843**. |
| 3 | Pairwise vs. rubric-based scoring | The COLM 2025 result: a content-neutral distractor flips **~35%** of pairwise verdicts vs **~9%** absolute. Default to absolute for verifiable criteria. |
| 4 | Known biases and how to mitigate them | Position, verbosity, self-enhancement, provenance, distracted, intransitivity — and the mitigation for each. |
| 5 | Panels and juries: ablate across judges, then vote | Ablate across ≥2 judge families; a PoLL jury of small diverse models beats one big judge, with less bias and **>7× cheaper**. |
| 6 | Calibration: turning a judge into a trustworthy instrument | Collect human corrections → build few-shot → track agreement. Use the cheapest capable evaluator; close the data flywheel. |
| 7 | Putting it together: RAG evaluation end to end | Score answer-correctness with an absolute, anchored 1–5 judge; sweep configs and measure. There is no single recipe. |
| 8 | Frameworks: you don't have to build this from scratch | RAGAS (faithfulness, answer relevancy, context precision/recall) and DeepEval (G-Eval, DAG, ArenaGEval). |
| 9 | Evaluating agents, not just outputs | Vocabulary, three grader families, outcome over trajectory, partial credit, **pass@k vs pass^k**, reward hacking and broken tasks. |
| 10 | The framework & platform landscape: who uses what | How LangChain, Microsoft, Anthropic, and OpenAI package tracing → datasets → graders → offline/online runs. |
| 11 | Agent benchmarks worth knowing | SWE-bench, Terminal-Bench, τ-bench, BFCL, ToolBench, GAIA, AgentBench, WebArena, OSWorld, BrowseComp — plus saturation and harness-sensitivity cautions. |
| 12 | Open-source agent frameworks: LangChain, LangGraph, Deep Agents | `agent = model + harness`; most behavior and headroom live in the harness (a harness-only change moved a fixed model **+13.7** on Terminal-Bench 2.0). |
| 13 | Worked examples: from a Deep Agent to an eval-driven harness optimizer | 13.1 the LLM Wiki (a Deep Agent worth evaluating); 13.2 better-harness (an agent that optimizes another agent's harness against a private holdout). |
| — | A working checklist | The end-to-end checklist, dataset → judge → protocol → jury → calibration → agents → harness. |
| — | References | The source corpus the guide synthesizes (see below). |

## Headline results (all reproduced in this repo)

| Claim | Figure |
| --- | --- |
| Strong judges vs. humans on MT-Bench | ≈ 80% agreement |
| Two-human inter-rater correlation (the ceiling) | ≈ 0.563 |
| Naive 0–10 judge → improved anchored 1–4 judge | 0.567 → 0.843 |
| Distracted-evaluation verdict flips: pairwise vs. absolute | ~35% vs ~9% |
| Tie recognition on equal pairs: absolute vs. pairwise | 84–93% vs 2–7% |
| PoLL jury vs. a single GPT-4-class judge | better agreement, > 7× cheaper |
| `pass^3` at a 75% per-trial rate | ≈ 42% |
| SWE-bench Verified, capability growth in ~1 year | ~30% → > 80% |
| Harness-only change on Terminal-Bench 2.0 (model fixed) | 52.8 → 66.5 (+13.7) |
| Curated skills vs. none (task completion) | ~9% → ~82% |

## Sources the guide synthesizes

- Aymeric Roucher — *RAG Evaluation* and *Using LLM-as-a-judge*, Hugging Face Cookbooks.
- LangChain — *How to Calibrate LLM-as-a-Judge with Human Corrections*.
- Tripathi, Wadhwa, Durrett, Niekum — *Pairwise or Pointwise?*, COLM 2025.
- Verga et al. — *Replacing Judges with Juries* (PoLL), 2024.
- RAGAS and DeepEval framework documentation.
- Anthropic — *Demystifying Evals for AI Agents* and *Building Effective Agents*.
- LangChain LangSmith / OpenEvals / AgentEvals; Microsoft AutoGen / Agent Framework; OpenAI Evals API.
- Deep Agents; LangChain — *Improving Deep Agents with Harness Engineering* and *Evaluating Deep Agents*; the **better-harness** example.

A full, linked reference list is at the end of the PDF and in the repo's
[top-level README](../README.md#references).

## Reading the guide alongside the code

Every part is turned into runnable, tested code. Read a section, then run its
example:

- **Part → module → example** map: [top-level README](../README.md#the-map-guide--code--example)
- **Concept → code** deep dive and the working checklist: [CONCEPTS.md](./CONCEPTS.md)
- Run a single part, e.g. Part 3: `python examples/03_pairwise_vs_rubric.py`
- Run everything: `python examples/run_all.py` · test everything: `python -m pytest`

## Version note

The PDF in this folder is the **v2** edition (31 pages). It is a superset of the
original 25-page edition: Parts 1–11 are unchanged, and v2 adds **Part 12**
(open-source agent frameworks) and **Part 13** (the two worked examples), two
checklist items on harness optimization, and the corresponding references.
