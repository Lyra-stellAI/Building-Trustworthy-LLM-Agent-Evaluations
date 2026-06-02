# Building Trustworthy LLM & Agent Evaluations

Runnable companion code for the practitioner's guide of the same name — a tour of
LLM-as-a-judge and agent evaluation: building, validating, and operationalizing
evaluators, and the methodologies the major labs actually use. It closes with the
open-source agent stack (LangChain, LangGraph, Deep Agents) and two worked
examples — an LLM-wiki Deep Agent and an eval-driven harness optimizer.

Every idea in the guide is turned into code you can **run, read, and test**. It
all runs **offline** against a deterministic *simulated judge* that reproduces the
documented phenomena (distracted-evaluation flip rates, position/verbosity bias,
jury effects, pass@k vs pass^k), so you can *see* the effects without an API key —
then swap in a real model when you're ready.

📄 The source guide (31-page PDF) and a reading guide live in [`docs/`](docs/README.md).

```bash
git clone <this-repo> && cd Building-Trustworthy-LLM-Agent-Evaluations
python -m pytest             # 74 tests, runs in ~1s, zero dependencies
python examples/run_all.py   # 13 illustrated walkthroughs, one per part
```

> **Is the simulator cheating?** No — and it's worth being precise about what's
> real. The *harness* is real and reusable: prompt templates, score parsing,
> jury aggregation, the statistics (correlation, Cohen's κ, pass@k/pass^k),
> trajectory matchers, partial credit. The *judge* is simulated so the
> experiments are deterministic and free: it models documented judge behavior
> (e.g. the COLM 2025 finding that comparison amplifies distractor sensitivity)
> off latent attributes instead of reading the prompt. Point the same harness at
> `AnthropicJudge`/`OpenAIJudge` for a genuine evaluation. See
> [The simulated judge](#the-simulated-judge).

---

## Why evaluation needs this much care

Surface metrics (ROUGE/BLEU) miss almost everything that matters once a task is
open-ended. LLM-as-a-judge scales, but **it does not work out of the box**: a
naive judge is unreliable, the way you frame the question introduces systematic
bias, and prompt-tweaking alone won't get you to an evaluator you'd trust for a
shipping decision. A useful framing throughout: *a judge is a scoring mechanism,
and like any mechanism it can be gamed.* This repo walks the full path — build a
dataset, build a judge, validate it, choose the right protocol, mitigate biases,
use a jury, calibrate, operationalize, and evaluate agents.

## The map: guide → code → example

| Part | Module | Example | Headline result (reproduced) |
| --- | --- | --- | --- |
| 1 · Synthetic dataset | [`datasets.py`](trustworthy_evals/datasets.py) | [`01`](examples/01_synthetic_dataset.py) | Critique agents discard ~half the candidates |
| 2 · Build & validate the judge | [`judge.py`](trustworthy_evals/judge.py) | [`02`](examples/02_build_and_validate_judge.py) | Naive **0.57** → improved **0.85** correlation |
| 3 · Pairwise vs. rubric | [`protocols.py`](trustworthy_evals/protocols.py) | [`03`](examples/03_pairwise_vs_rubric.py) | Distractor flips **~37%** pairwise vs **~6%** absolute |
| 4 · Biases & mitigations | [`biases.py`](trustworthy_evals/biases.py) | [`04`](examples/04_biases.py) | Position/verbosity/self-enhancement shrink under mitigation |
| 5 · Panels & juries | [`jury.py`](trustworthy_evals/jury.py) | [`05`](examples/05_jury.py) | Diverse panel **κ 0.64 vs 0.30**, ~7× cheaper |
| 6 · Calibration | [`calibration.py`](trustworthy_evals/calibration.py) | [`06`](examples/06_calibration.py) | Agreement climbs **0.66 → 0.98** over the loop |
| 7 · RAG end to end | [`rag_eval.py`](trustworthy_evals/rag_eval.py) | [`07`](examples/07_rag_end_to_end.py) | Config sweep: chunk-size is high-impact; no single recipe |
| 8 · Frameworks | [`frameworks.py`](trustworthy_evals/frameworks.py) | [`08`](examples/08_frameworks.py) | RAGAS metrics + DeepEval G-Eval/DAG/ArenaGEval |
| 9 · Evaluating agents | [`agent_eval.py`](trustworthy_evals/agent_eval.py) | [`09`](examples/09_agent_eval.py) | Outcome grading, partial credit, **pass@10≈1.0 vs pass^10≈0.03** |
| 10–11 · Landscape & benchmarks | [`benchmarks.py`](trustworthy_evals/benchmarks.py) | [`10`](examples/10_benchmarks.py) | Benchmark registry + per-agent playbooks |
| 12 · agent = model + harness | [`harness.py`](trustworthy_evals/harness.py) | [`11`](examples/11_harness_engineering.py) | Harness-only change **+13.7** on Terminal-Bench; skills **9%→82%** |
| 13 · Worked examples | [`deep_agent.py`](trustworthy_evals/deep_agent.py), [`better_harness.py`](trustworthy_evals/better_harness.py) | [`12`](examples/12_deep_agent_wiki.py), [`13`](examples/13_better_harness.py) | LLM Wiki graded by outcome; optimizer rejects overfits, **scorecard 0→2** |
| — · Statistics | [`metrics.py`](trustworthy_evals/metrics.py) | — | Pearson, Cohen's κ, pass@k / pass^k |
| — · Judge & prompts | [`llm.py`](trustworthy_evals/llm.py), [`prompts.py`](trustworthy_evals/prompts.py) | — | SimulatedJudge + real adapters; verbatim templates |

A deeper concept-to-code map lives in [`docs/CONCEPTS.md`](docs/CONCEPTS.md).

## The two halves of an evaluation pipeline

Every setup needs (1) an evaluation dataset and (2) an evaluator. LLMs can help
with both:

**Part 1 — manufacture a dataset and filter it with critique agents.**

```python
from trustworthy_evals.datasets import candidate_qa_pairs, filter_eval_set

report = filter_eval_set(candidate_qa_pairs())   # groundedness / relevance / standalone, keep >= 4/5
print(f"{report.kept}/{report.total} survived ({report.keep_rate:.0%})")  # ~half discarded
```

**Part 2 — build the judge, then *measure it against humans before trusting it*.**

```python
from trustworthy_evals.judge import simulate_validation_study

study = simulate_validation_study(seed=0)
sub = study.agreement_subset()                  # clean labels: where two annotators agree
print(study.inter_rater())                       # ~0.56  <- the CEILING (noisy ground truth)
print(study.naive_correlation(sub))              # ~0.55  <- barely above the ceiling
print(study.improved_correlation(sub))           # ~0.85  <- anchored rubric + reasoning-first
```

## The headline finding: pairwise is the fragile protocol (Part 3)

The conventional wisdom says pairwise comparison is more reliable than absolute
scoring. For verifiable / correctness-oriented tasks it's **backwards** — the
COLM 2025 result. A content-neutral distractor injected into the *worse* answer
flips the verdict far more often under pairwise:

```text
--- Verdict flip rate when a distractor is added to the worse answer ---
  distractor       pairwise   absolute
  assertiveness      44.1%       5.5%
  prolixity          38.4%       7.7%
  sycophancy         27.5%       4.0%
  OVERALL            36.7%       5.7%   (paper: ~35% vs ~9%)

--- Leaderboard hacking: rewrite the bottom models to be assertive (no facts changed) ---
  pairwise rank before : ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot']
  pairwise rank after  : ['delta', 'echo', 'alpha', 'foxtrot', 'bravo', 'charlie']   # delta: #4 -> #1
  absolute rank before : ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot']
  absolute rank after  : ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot']   # unchanged
```

**Practical rule:** for verifiable criteria or near-tie-heavy data, default to
*absolute* scoring; reserve *pairwise* for open-ended quality with no anchor, and
then run both orders and guard against distracted evaluation.

## Evaluating agents, not just outputs (Part 9)

Agents are systems — they plan, call tools, keep state, and only eventually
return something. Grade the **outcome** over the trajectory, mix grader families,
build in **partial credit**, and report a **rate**, not a verdict.

```python
from trustworthy_evals.agent_eval import auth_bypass_task, make_flaky_agent, run_trials
from trustworthy_evals.metrics import pass_at_k, pass_hat_k

report = run_trials(auth_bypass_task(), make_flaky_agent(0.75), k=10)
print(pass_at_k(report.observed_rate, 10))   # ~1.0  "at least one success" (rises with k)
print(pass_hat_k(report.observed_rate, 10))  # ~0.03 "succeeds every time"  (falls with k)
```

At `k=1` they're identical; by `k=10` they tell opposite stories. Reliability-
critical deployments watch `pass^k`.

## agent = model + harness (Parts 12–13)

Most of an agent's behavior — and most of your headroom to improve it — lives in
the **harness** (prompt, tools, skills, context management, middleware), not the
frozen weights. Changing *only* the harness moved a fixed model **+13.7 points**
on Terminal-Bench, and curated skills lifted task completion from **9% to 82%**:

```python
from trustworthy_evals.harness import harness_engineering_demo, skills_ablation_demo
print(harness_engineering_demo())   # 52.8 -> 66.5  (+13.7), model held fixed
print(skills_ablation_demo())       # 9% -> 82% task completion
```

The two worked examples make it concrete. The **LLM Wiki** is a Deep Agent worth
evaluating — graded by *outcome* (did the index refresh? did a declined review
skip writes?), *groundedness* of cited answers, and a code grader over its
machine-parseable log. **better-harness** is an eval-driven optimizer where one
agent edits another's harness and a change is kept only if it generalizes to a
*private holdout* — the Part 3 mechanism-design thesis made literal:

```text
--- better-harness decision log (the outer agent sees only TRAIN failures) ---
  it0 KEEP generalizing  raise[conversation]                train=3 hold=2  accepted
  it0 drop overfitting   memorize[visible train failures]   train=6 hold=0  holdout regressed (did not generalize)
  ...
  train 2->6   holdout 2->4 (private)   scorecard 0->2 (untouched -> the honest report)
```

The overfitting "memorize the visible cases" hack always maxes the training
score and is **rejected every iteration** because it regresses the hidden
holdout. The structure, not the prompt, guarantees the only way to score is to
genuinely improve the harness.

## The simulated judge

`SimulatedJudge` models a response as a latent `quality` plus style features
(assertiveness, prolixity, sycophancy, length, source family). Each bias is an
**explicit, documented knob** whose default sits near the literature's central
estimate — so the bundled demos reproduce the guide's numbers, and setting a knob
to `0` "mitigates" that bias so you can watch the effect vanish.

```python
from trustworthy_evals.llm import Response, SimulatedJudge

judge = SimulatedJudge()
good = Response(text="A precise, correct answer.", quality=0.8)
print(judge.score_absolute(good, scale=(1, 5)))          # rubric/pointwise score
print(judge.compare_both_orders(good, Response("...", quality=0.5)))  # pairwise, position-bias-cancelled
```

To run a **real** evaluation, the same harness accepts a real client:

```python
from trustworthy_evals.llm import AnthropicJudge, OpenAIJudge
from trustworthy_evals.judge import Judge

judge = Judge(AnthropicJudge("claude-haiku-4-5-20251001"))  # pip install anthropic + ANTHROPIC_API_KEY
print(judge.score(question="What is RAG?", answer="Retrieval-augmented generation."))
```

## Project layout

```
trustworthy_evals/      # the library (pure standard library, no runtime deps)
  prompts.py            # verbatim prompt templates from the guide
  llm.py                # SimulatedJudge + Response + real LLM adapters
  metrics.py            # pearson, cohens_kappa, pass_at_k, pass_hat_k, ...
  datasets.py judge.py protocols.py biases.py jury.py
  calibration.py rag_eval.py frameworks.py agent_eval.py benchmarks.py
  harness.py deep_agent.py better_harness.py   # Parts 12-13
examples/               # 13 runnable, illustrated walkthroughs (+ run_all.py)
tests/                  # 74 pytest tests asserting the documented relationships
docs/CONCEPTS.md        # concept-to-code map and the working checklist
```

## Installation (optional)

The repo runs straight from a clone (the examples bootstrap `sys.path`, and
`pytest` uses a root `conftest.py`). For an editable install:

```bash
pip install -e ".[dev]"        # + pytest
pip install -e ".[anthropic]"  # real Anthropic judge
pip install -e ".[openai]"     # real OpenAI judge
```

## References

The guide synthesizes, and this code illustrates:

- Aymeric Roucher — *RAG Evaluation* & *Using LLM-as-a-judge*, Hugging Face Cookbooks (synthetic datasets, critique agents, the 0.567 → 0.843 prompt story).
- *How to Calibrate LLM-as-a-Judge with Human Corrections* — LangChain (judge types, bias taxonomy, the calibration loop, the data flywheel).
- Tripathi, Wadhwa, Durrett, Niekum — *Pairwise or Pointwise?*, COLM 2025 (distracted evaluation; ~35% vs ~9% flips; tie rejection; leaderboard hacking; intransitivity). [code](https://github.com/UMass-SCALAR-Lab/distracted_evaluation)
- Verga et al. — *Replacing Judges with Juries* (PoLL), 2024. [arXiv:2404.18796](https://arxiv.org/abs/2404.18796)
- [RAGAS](https://docs.ragas.io) · [DeepEval](https://deepeval.com) — component and CI-style metric frameworks.
- Anthropic — [*Demystifying Evals for AI Agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (grader families, capability vs. regression, pass@k/pass^k, agent vs. eval harness, eval-driven development).
- LangChain LangSmith / OpenEvals / AgentEvals; Microsoft AutoGen / Agent Framework; OpenAI Evals API — the platform landscape (Part 10).
- [Deep Agents](https://github.com/langchain-ai/deepagents) — the batteries-included agent harness on LangGraph (planning, virtual filesystem, sub-agents, middleware, pluggable backends).
- LangChain — *Improving Deep Agents with Harness Engineering* (2026): `agent = model + harness`; harness-only changes moved `deepagents-cli` +13.7 pts (52.8 → 66.5) on Terminal-Bench 2.0.
- LangChain — *Evaluating Deep Agents* (2025): per-case success criteria and trajectory/state assertions; curated skills lifted task completion from ~9% to ~82%.
- [better-harness](https://github.com/langchain-ai/deepagents/tree/main/examples/better-harness) — a Deep Agent that optimizes another agent's harness against train/holdout/scorecard splits (kin to karpathy's `autoresearch` and Stanford's Meta-Harness).

> The code is an educational reimplementation for the tutorial. For production,
> reach for the real frameworks (RAGAS, DeepEval) and platforms (LangSmith,
> Braintrust, Langfuse, Phoenix, Harbor) referenced above.
