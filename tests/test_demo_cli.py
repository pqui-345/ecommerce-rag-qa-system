import json
from pathlib import Path

from src.demo_cli import (
    load_jsonl_records,
    main,
    search_bm25_chunks,
    summarize_test_records,
)


def test_load_jsonl_records_handles_empty_and_corrupt_lines(tmp_path: Path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    assert load_jsonl_records(empty_file) == []
    assert load_jsonl_records(tmp_path / "missing.jsonl") == []

    mixed_file = tmp_path / "mixed.jsonl"
    mixed_file.write_text(
        '{"qid_moi": "q1"}\n{broken json\n\n{"qid_moi": "q2"}\n',
        encoding="utf-8",
    )
    rows = load_jsonl_records(mixed_file)
    assert [r["qid_moi"] for r in rows] == ["q1", "q2"]


def test_search_bm25_chunks_scoped_and_global():
    chunks = [
        {"asin": "A1", "chunk_id": "c1", "text": "Works only with 110 volt power adapter"},
        {"asin": "A1", "chunk_id": "c2", "text": "USB cable length is 2 meters"},
        {"asin": "A2", "chunk_id": "c3", "text": "Universal 110 volt and 220 volt adapter"},
    ]
    hits_scoped, ms_scoped = search_bm25_chunks(chunks, "110 volt adapter", top_k=2, asin="A1")
    assert ms_scoped >= 0.0
    assert len(hits_scoped) == 2
    assert hits_scoped[0][0]["chunk_id"] == "c1"
    assert all(item[0]["asin"] == "A1" for item in hits_scoped)

    hits_global, _ = search_bm25_chunks(chunks, "220 volt", top_k=1)
    assert hits_global[0][0]["chunk_id"] == "c3"


def test_summarize_test_records_empty_and_populated():
    empty = summarize_test_records([])
    assert empty["total"] == 0
    assert empty["hallucination_rate_pct"] == 0.0

    records = [
        {"is_abstention": False, "judge_answerable": 1, "faithfulness": 1.0, "context_precision": 0.8, "answer_relevance": 1.0, "has_hallucination": 0},
        {"is_abstention": False, "judge_answerable": 0, "faithfulness": 0.0, "context_precision": 0.0, "answer_relevance": 0.5, "has_hallucination": 1},
        {"is_abstention": True, "judge_answerable": 0, "faithfulness": 1.0, "context_precision": 0.0, "answer_relevance": 0.0, "has_hallucination": 0},
    ]
    stats = summarize_test_records(records)
    assert stats["total"] == 3
    assert stats["direct"] == 2
    assert stats["hallucinations"] == 1
    assert stats["hallucination_rate_pct"] == 50.0
    assert stats["faithfulness_direct"] == 0.5
    assert stats["cp5_answerable"] == 0.8


def test_demo_cli_step_1_runs_cleanly(capsys):
    rc = main(["--step", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Luy dang x2 : True" in out
    assert "Bao toan '5*3' & '<10 hours': True" in out
