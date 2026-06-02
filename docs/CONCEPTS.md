# Concepts → code

A reference mapping every idea in the guide to the code that implements it, plus
the working checklist at the end.

## Part 1 — Building a synthetic evaluation dataset
- **Idea:** sample corpus chunks → generate factoid QA → filter with three
  critique agents (groundedness, relevance, standalone-ness); keep ≥ 4/5 on all.
- **Code:** `datasets.candidate_qa_pairs`, `datasets.critique_qa`,
  `datasets.filter_eval_set`; prompts `QA_GENERATION_PROMPT`, `CRITIQUE_PROMPT`.
- **Takeaway:** filters are aggressive (~half discarded). Generate 200+ for 100+
  survivors. Each rejected pair fails exactly the criterion its flaw targets.

## Part 2 — Building the judge (and validating it first)
- **Idea:** never ship an unvalidated judge. Establish a two-human baseline (the
  ceiling), then improve the prompt: reasoning-before-score, anchored integer
  scale, reference answer, structured output.
- **Code:** `judge.extract_score` (robust parsing), `judge.Judge` (real path),
  `judge.simulate_validation_study` (reproduces 0.563 / 0.567 / 0.843).
- **Takeaway:** the human ceiling caps judge quality; cleaning to the agreement
  subset raises the achievable correlation. You'll never hit 1.0.

## Part 3 — Pairwise vs. rubric-based scoring
- **Idea:** pairwise asks "which is better?"; rubric asks "how good, on a scale?".
  For verifiable criteria, absolute is *more* robust — pairwise is swayed by
  content-neutral distractors, refuses ties, and is gameable.
- **Code:** `protocols.run_distracted_experiment`, `protocols.summarize_distracted`,
  `protocols.tie_recognition_experiment`, `protocols.leaderboard_hacking_demo`.
- **Takeaway:** default to absolute for correctness/near-ties; reserve pairwise
  for anchorless open-ended quality, then run both orders.

## Part 4 — Known biases and how to mitigate them
- **Idea:** position, verbosity, self-enhancement, provenance, distracted,
  intransitivity. Instruction helps but isn't a complete fix.
- **Code:** `biases.BIAS_TABLE`, `biases.position_bias_demo`,
  `biases.verbosity_bias_demo`, `biases.self_enhancement_demo`; the `mitigate=`
  flag and both-orders aggregation on `SimulatedJudge`.
- **Takeaway:** different judge family than the generator; tell it to ignore
  length/position/names; calibrate against your domain.

## Part 5 — Panels and juries
- **Idea:** a single judge is a single point of failure (judge choice decides
  results; self-preference bias). A jury of small, diverse models (PoLL) agrees
  with humans better, is less biased, and is cheaper.
- **Code:** `jury.jury_verdict`, `jury.Jury`, `jury.panel_vs_single_demo`,
  `jury.judge_ablation`.
- **Takeaway:** odd number of diverse judges; majority vote for binary, average
  pooling for graded; route disagreement to a human; always ablate across ≥ 2
  families.

## Part 6 — Calibration
- **Idea:** prompt iteration alone won't close the gap. Loop: collect human
  corrections → build few-shot → track agreement over time. Use the cheapest
  capable evaluator (rules → metrics → judge → human).
- **Code:** `calibration.run_calibration_loop`, `calibration.choose_evaluator`,
  `calibration.SamplingPolicy`, `calibration.DATA_FLYWHEEL`.
- **Takeaway:** offline evals for regression, online (sampled) for drift,
  thread-level scoring for agents; close the flywheel.

## Part 7 — RAG evaluation end to end
- **Idea:** many knobs (chunk size, embeddings, rerank, reader); tuning is
  pointless without measurement. Score answer-correctness with an absolute,
  anchored 1-5 judge that has the reference answer and a `[RESULT]` marker.
- **Code:** `rag_eval.KeywordRetriever`, `rag_eval.run_rag`,
  `rag_eval.score_with_rag_judge`, `rag_eval.sweep_configs`; `RAG_EVAL_PROMPT`.
- **Takeaway:** no single recipe; chunk-size tends to be cheap and high-impact;
  measure *your* system.

## Part 8 — Frameworks
- **Idea:** don't hand-roll. RAGAS = RAG component metrics (mostly LLM-judged
  decomposition); DeepEval = pytest for LLM outputs (G-Eval, DAG, ArenaGEval).
- **Code:** `frameworks.faithfulness/answer_relevancy/context_precision/
  context_recall/ragas_score`; `frameworks.GEval/DAGMetric/ArenaGEval/
  LLMTestCase/assert_test`.
- **Takeaway:** RAGAS for tuning retriever+generator; DeepEval for CI, custom
  rubric metrics, and deterministic decision-tree scoring.

## Part 9 — Evaluating agents, not just outputs
- **Idea:** agents are systems. Vocabulary (task/trial/grader/transcript/
  outcome/harness). Three grader families (code/model/human). Outcome over
  trajectory. Partial credit. Report a rate (pass@k vs pass^k). Watch for reward
  hacking and broken tasks.
- **Code:** `agent_eval.Task/Transcript/grade_task`, the grader classes,
  `agent_eval.trajectory_match`, `agent_eval.run_trials`,
  `agent_eval.auth_bypass_task`, `agent_eval.classify_eval_health`,
  `agent_eval.likely_broken_task`.
- **Takeaway:** prefer deterministic graders; LLM rubric for nuance; humans to
  calibrate; isolate trials; a 0% pass@100 usually means a broken task.

## Parts 10–11 — Landscape & benchmarks
- **Code:** `benchmarks.BENCHMARKS`, `benchmarks.LEADERBOARD_CAUTIONS`,
  `agent_eval.AGENT_PLAYBOOKS`.
- **Takeaway:** an agentic score measures harness + model + environment; compare
  like-for-like; build your own task bank from real failures.

---

## The working checklist (from the guide)

- [ ] **Dataset:** synthetic QA, filtered on groundedness/relevance/standalone
  (≥ 4/5); 100+ survivors → generate 200+.
- [ ] **Validate first:** ~30 examples, two annotators, report inter-rater as the
  ceiling; clean by averaging or agreement-only.
- [ ] **Judge prompt:** anchored integer scale, reasoning before score, reference
  if available, structured output.
- [ ] **Protocol:** absolute for verifiable/low-preference-strength data;
  pairwise only without an anchor, then both orders.
- [ ] **Robustness:** ablate across ≥ 2 judge families; jury for important calls
  (majority for binary, average for graded; odd count; route disagreement).
- [ ] **Bias defense:** different judge family than generator; ignore
  length/position/names; calibrate to your domain.
- [ ] **Calibrate continuously:** corrections → few-shot → tracked agreement.
- [ ] **Operate:** offline regression + online (sampled) drift; thread-level for
  agents; close the flywheel.
- [ ] **Tooling:** RAGAS for RAG metrics; DeepEval for CI / G-Eval / DAG.
- [ ] **Agents:** grade the outcome; partial credit; layer transcript checks only
  where you care; report pass@1 or pass^k; isolated trials.
- [ ] **Capability vs. regression:** low-pass capability hills + ~100% regression
  guards; watch for saturation.
- [ ] **Measure your real system:** there is no universal recipe.
