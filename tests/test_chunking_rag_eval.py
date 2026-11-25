"""Kiểm thử cho module sinh câu trả lời và chấm điểm RAGAS đối sánh chiến lược phân đoạn."""

import json
from pathlib import Path

from src.chunking_rag_eval import chay_rag_va_judge_cac_chien_luoc


def _fake_llm_call(messages, model_name):
    user_text = messages[-1]["content"]
    if "waterproof" in user_text.lower():
        return (
            "The provided product reviews do not contain enough information.",
            {"latency_sec": 0.1, "prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        )
    return (
        "Yes, it supports 1080p Full HD via HDMI.",
        {"latency_sec": 0.1, "prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    )


def _fake_gemini_call(prompt_text, model_name):
    return [
        {
            "qid_moi": "q-1",
            "chunk_relevance": [1, 0, 0, 0, 0],
            "judge_answerable": 1,
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "has_hallucination": 0,
        },
        {
            "qid_moi": "q-2",
            "chunk_relevance": [0, 0, 0, 0, 0],
            "judge_answerable": 0,
            "faithfulness": 1.0,
            "answer_relevance": 0.0,
            "has_hallucination": 0,
        },
        {
            "qid_moi": "q-3",
            "chunk_relevance": [1, 1, 0, 0, 0],
            "judge_answerable": 1,
            "faithfulness": 1.0,
            "answer_relevance": 0.9,
            "has_hallucination": 0,
        },
    ]


def test_chay_rag_va_judge_cac_chien_luoc(tmp_path: Path):
    rows = [
        {
            "qid_moi": "q-1",
            "asin": "A1",
            "doc_id": "doc-A1",
            "question_text_clean": "Does it support 1080p?",
            "nhan_chot_3_muc": "yes",
            "nhan_chot_chi_tiet": "answerable_full",
            "reference_answer": "Yes 1080p.",
            "top_k_scoped": [{"chunk_id": "c1", "text": "Supports 1080p HDMI."}],
        },
        {
            "qid_moi": "q-2",
            "asin": "A1",
            "doc_id": "doc-A1",
            "question_text_clean": "Is it waterproof?",
            "nhan_chot_3_muc": "no",
            "nhan_chot_chi_tiet": "unanswerable_in_doc",
            "reference_answer": "",
            "top_k_scoped": [{"chunk_id": "c1", "text": "Supports 1080p HDMI."}],
        },
        {
            "qid_moi": "q-3",
            "asin": "A1",
            "doc_id": "doc-A1",
            "question_text_clean": "How long does the battery last?",
            "nhan_chot_3_muc": "partial",
            "nhan_chot_chi_tiet": "partial_expanded",
            "reference_answer": "About 10 hours.",
            "top_k_scoped": [{"chunk_id": "c2", "text": "Battery lasts 10 hours."}],
        },
    ]
    for st in ("fixed", "structure"):
        p = tmp_path / f"retrieval_vector_dev_{st}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    report_path = tmp_path / "ragas_comp.md"
    res = chay_rag_va_judge_cac_chien_luoc(
        processed_dir=tmp_path,
        report_path=report_path,
        strategies=("fixed", "structure"),
        gen_delay_sec=0.0,
        judge_delay_sec=0.0,
        llm_call_fn=_fake_llm_call,
        gemini_call_fn=_fake_gemini_call,
    )
    assert report_path.exists()
    assert len(res["summaries"]) == 2
    s0 = res["summaries"][0]
    assert s0["n_direct"] == 2
    assert s0["trap_rejection_rate"] == 100.0
    assert s0["faithfulness_direct"] == 1.0
