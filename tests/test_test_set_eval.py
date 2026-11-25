"""Kiểm thử cho module đánh giá tập test độc lập src/test_set_eval.py."""

import json
from pathlib import Path

from src.chunking import tao_chunks_tu_file_docs
from src.test_set_eval import (
    chay_danh_gia_chot_tap_test,
    chuan_bi_truy_xuat_test,
    lay_mau_phan_tang_test_120,
)


def test_lay_mau_phan_tang_va_chay_test_eval(tmp_path: Path):
    q_rows = []
    idx = 0
    for qt in ("descriptive", "yesno"):
        for ia in (0, 1):
            for _ in range(2):
                idx += 1
                q_rows.append(
                    {
                        "qid_moi": f"q-test-{idx:03d}",
                        "asin": "T1",
                        "doc_id": "doc-T1",
                        "question_type": qt,
                        "is_answerable_auto": ia,
                        "question_text_clean": "Does this support 1080p HDMI?",
                        "reference_answer": "Yes, 1080p HDMI.",
                        "review_snippets_clean": ["Supports 1080p HDMI output clearly."],
                    }
                )
    q_path = tmp_path / "questions_test_clean.jsonl"
    q_path.write_text("\n".join(json.dumps(r) for r in q_rows) + "\n", encoding="utf-8")

    eval_p = tmp_path / "eval_test.jsonl"
    res_s = lay_mau_phan_tang_test_120(q_path, eval_p, n_per_cell=1, seed=42)
    assert res_s["n_selected"] == 4

    docs_p = tmp_path / "docs_test_clean.jsonl"
    docs_p.write_text(
        json.dumps(
            {
                "doc_id": "doc-T1",
                "asin": "T1",
                "category": "Electronics",
                "nguon_goc": "test",
                "text_clean": "Supports 1080p HDMI output clearly.\n\nBattery lasts 8 hours on charge.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tao_chunks_tu_file_docs(docs_p, tmp_path / "chunks_test_fixed.jsonl", chunk_size_words=10, chunk_overlap_words=2)

    fake_emb = lambda texts: [[1.0, 0.0] if "1080p" in t or "hdmi" in t.lower() else [0.0, 1.0] for t in texts]
    res_ret = chuan_bi_truy_xuat_test(
        processed_dir=tmp_path,
        eval_test_path=eval_p,
        report_path=tmp_path / "test_ret.md",
        embed_docs_fn=fake_emb,
        embed_queries_fn=fake_emb,
        top_k=2,
    )
    assert len(res_ret["summary"]) == 3
    assert (tmp_path / "test_ret.md").exists()

    def fake_llm(msgs, m):
        if "Retrieved Product Review" in msgs[-1]["content"]:
            return "Yes, it supports 1080p HDMI.", {"total_tokens": 20}
        return "I do not have enough information.", {"total_tokens": 10}

    def fake_gem(prompt, m):
        return [
            {
                "qid_moi": f"q-test-{i:03d}",
                "chunk_relevance": [1, 0, 0, 0, 0],
                "judge_answerable": 1,
                "faithfulness": 1.0,
                "answer_relevance": 1.0,
                "has_hallucination": 0,
            }
            for i in range(1, 9)
        ]

    res_j = chay_danh_gia_chot_tap_test(
        processed_dir=tmp_path,
        eval_test_path=eval_p,
        report_path=tmp_path / "test_final.md",
        llm_call_fn=fake_llm,
        gemini_call_fn=fake_gem,
    )
    assert len(res_j["summaries"]) == 4
    assert (tmp_path / "test_final.md").exists()
