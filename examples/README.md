# Examples

Ten runnable scripts, one per part of the guide. Each prints a clear,
self-contained illustration and ends with the one-line lesson. They run offline
against the deterministic `SimulatedJudge` — no API keys, no network.

```bash
# from the repo root
python examples/run_all.py            # run all ten in order
python examples/03_pairwise_vs_rubric.py   # or run a single part
```

| Script | Part | Illustrates |
| --- | --- | --- |
| `01_synthetic_dataset.py` | 1 | Critique agents filter ~half the generated QA pairs |
| `02_build_and_validate_judge.py` | 2 | Naive 0.57 → improved 0.84 correlation; the human ceiling |
| `03_pairwise_vs_rubric.py` | 3 | ~35% pairwise vs ~9% absolute flips; tie rejection; leaderboard hacking |
| `04_biases.py` | 4 | Position / verbosity / self-enhancement bias, with mitigations |
| `05_jury.py` | 5 | A diverse panel beats one big judge (κ), 7× cheaper; judge ablation |
| `06_calibration.py` | 6 | Agreement climbs as corrections become few-shot examples |
| `07_rag_end_to_end.py` | 7 | Config sweep: chunk-size is high-impact, no single recipe |
| `08_frameworks.py` | 8 | RAGAS metrics; DeepEval G-Eval / DAG / ArenaGEval |
| `09_agent_eval.py` | 9 | Outcome grading, partial credit, pass@k vs pass^k, reward hacking |
| `10_benchmarks.py` | 10–11 | Benchmark registry, per-agent playbooks, leaderboard cautions |
| `11_harness_engineering.py` | 12 | `agent = model + harness`; +13.7 Terminal-Bench; skills 9%→82% |
| `12_deep_agent_wiki.py` | 13.1 | LLM Wiki graded by outcome, groundedness, and its log |
| `13_better_harness.py` | 13.2 | Eval-driven optimizer; overfits rejected, holdout/scorecard rise |

## Using a real model instead of the simulator

The harness code (prompt templates, score parsing, jury aggregation, metrics,
trajectory matching) is real and model-agnostic. To run a genuine evaluation,
construct a real judge and pass it where an `LLMClient` is expected:

```python
from trustworthy_evals.llm import AnthropicJudge
from trustworthy_evals.judge import Judge

judge = Judge(AnthropicJudge(model="claude-haiku-4-5-20251001"))  # needs `pip install anthropic` + ANTHROPIC_API_KEY
print(judge.score(question="What is RAG?", answer="Retrieval-augmented generation."))
```
