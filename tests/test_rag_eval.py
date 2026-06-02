from trustworthy_evals.rag_eval import (
    EVAL_SET,
    RagConfig,
    chunk_document,
    run_rag,
    score_with_rag_judge,
    sweep_configs,
)


def test_chunk_document_sizes():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_document(text, 25)
    assert len(chunks) == 4
    assert all(len(c.split()) <= 25 for c in chunks)


def test_run_rag_produces_one_answer_per_item():
    runs = run_rag(RagConfig())
    assert len(runs) == len(EVAL_SET)
    score = score_with_rag_judge(runs)
    assert 0.0 <= score <= 1.0


def test_chunk_size_is_high_impact():
    # The tutorial's empirical point: chunk size is a big lever here.
    big = score_with_rag_judge(run_rag(RagConfig(chunk_size=40, reader="strong")))
    small = score_with_rag_judge(run_rag(RagConfig(chunk_size=12, reader="strong")))
    assert abs(big - small) > 0.1


def test_sweep_orders_results_and_strong_reader_wins():
    results = sweep_configs()
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # The top configuration uses the strong reader.
    assert results[0].config.reader == "strong"
