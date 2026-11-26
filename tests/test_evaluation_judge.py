"""Kiểm thử cho module Giám khảo đánh giá RAGAS src/evaluation_judge.py."""

import json
from pathlib import Path
from typing import List, Union

from src.evaluation_judge import (
    chay_danh_gia_judge_baseline,
    normalize_judge_item,
    tao_prompt_cham_batch_no_rag_direct,
    tao_prompt_cham_batch_rag,
    tinh_context_precision_at_k,
)


def test_tinh_context_precision_at_k():
    assert tinh_context_precision_at_k([]) == 0.0
    assert tinh_context_precision_at_k([0, 0, 0, 0, 0]) == 0.0
    assert tinh_context_precision_at_k([1, 0, 0, 0, 0]) == 1.0
    assert tinh_context_precision_at_k([1, 1, 0, 0, 0]) == 1.0
    assert tinh_context_precision_at_k([1, 0, 1, 0, 0]) == 0.8333


def test_tao_prompt_cham_batch():
    rag_items = [
        {
            "qid_moi": "q1",
            "question_text_clean": "Does the stand swivel?",
            "reference_answer": "No.",
            "retrieved_contexts": ["The stand is fixed."],
            "generated_answer": "No, the stand is fixed.",
            "is_abstention": False,
        }
    ]
    p_rag = tao_prompt_cham_batch_rag(rag_items)
    assert "Does the stand swivel?" in p_rag
    assert "chunk_relevance" in p_rag

    norag_items = [
        {
            "qid_moi": "q1",
            "asin": "A1",
            "question_text_clean": "Does the stand swivel?",
            "nhan_chot_3_muc": "no",
            "reference_answer": "No.",
            "generated_answer": "Yes, it swivels 360 degrees.",
        }
    ]
    p_nr = tao_prompt_cham_batch_no_rag_direct(norag_items)
    assert "has_hallucination" in p_nr


def test_chay_danh_gia_judge_baseline_dong_hong_va_file_rong(tmp_path: Path):
    def _mock_gemini(prompt: str, model_name: str) -> Union[dict, List[dict]]:
        if "no_rag_generated_answer" in prompt:
            return [
                {
                    "qid_moi": "q-1",
                    "faithfulness": 0.0,
                    "answer_relevance": 0.0,
                    "has_hallucination": 1,
                }
            ]
        return [
            {
                "qid_moi": "q-1",
                "chunk_relevance": [1, 0, 1, 0, 0],
                "judge_answerable": 1,
                "faithfulness": 1.0,
                "answer_relevance": 0.95,
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
        ]

    eval_p = tmp_path / "eval.jsonl"
    nr_p = tmp_path / "norag.jsonl"
    rag_p = tmp_path / "rag.jsonl"
    out_nr = tmp_path / "out_nr.jsonl"
    out_rag = tmp_path / "out_rag.jsonl"
    rep_p = tmp_path / "rep.md"

    eval_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid_moi": "q-1",
                        "asin": "A1",
                        "question_text_clean": "Does it support USB?",
                        "nhan_chot_3_muc": "yes",
                        "nhan_chot_chi_tiet": "answerable_full",
                        "reference_answer": "Yes, USB 2.0.",
                    }
                ),
                "{dong json hong",
                json.dumps(
                    {
                        "qid_moi": "q-2",
                        "asin": "A2",
                        "question_text_clean": "Does the stand swivel?",
                        "nhan_chot_3_muc": "no",
                        "nhan_chot_chi_tiet": "unanswerable_in_doc",
                        "reference_answer": "No.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    nr_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid_moi": "q-1",
                        "asin": "A1",
                        "question_text_clean": "Does it support USB?",
                        "nhan_chot_3_muc": "yes",
                        "nhan_chot_chi_tiet": "answerable_full",
                        "is_abstention": False,
                        "generated_answer": "It is a 24-inch OLED TV.",
                    }
                ),
                json.dumps(
                    {
                        "qid_moi": "q-2",
                        "asin": "A2",
                        "question_text_clean": "Does the stand swivel?",
                        "nhan_chot_3_muc": "no",
                        "nhan_chot_chi_tiet": "unanswerable_in_doc",
                        "is_abstention": True,
                        "generated_answer": "I do not have enough information.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rag_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid_moi": "q-1",
                        "asin": "A1",
                        "question_text_clean": "Does it support USB?",
                        "nhan_chot_3_muc": "yes",
                        "nhan_chot_chi_tiet": "answerable_full",
                        "retrieved_contexts": ["Supports USB 2.0."],
                        "is_abstention": False,
                        "generated_answer": "Yes, it supports USB 2.0.",
                    }
                ),
                json.dumps(
                    {
                        "qid_moi": "q-2",
                        "asin": "A2",
                        "question_text_clean": "Does the stand swivel?",
                        "nhan_chot_3_muc": "no",
                        "nhan_chot_chi_tiet": "unanswerable_in_doc",
                        "retrieved_contexts": ["Great picture quality."],
                        "is_abstention": True,
                        "generated_answer": "The provided product reviews do not contain enough information.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kq = chay_danh_gia_judge_baseline(
        eval_labeled_path=eval_p,
        no_rag_gen_path=nr_p,
        rag_gen_path=rag_p,
        out_no_rag_judge_path=out_nr,
        out_rag_judge_path=out_rag,
        report_path=rep_p,
        gemini_call_fn=_mock_gemini,
    )
    assert kq["summary_no_rag"]["hallucination_count_total"] == 1
    assert kq["summary_rag"]["hallucination_count_total"] == 0
    assert kq["summary_rag"]["faithfulness_direct"] == 1.0
    assert kq["kappa_judge_vs_human_strict"]["kappa"] == 1.0
    assert rep_p.exists()


def test_normalize_judge_item_symmetric_fallback(tmp_path: Path):
    nr_fallback = normalize_judge_item({}, is_abstention=False, item_id="q-err-nr")
    rag_fallback = normalize_judge_item({}, is_abstention=False, item_id="q-err-rag")
    assert nr_fallback == rag_fallback
    assert nr_fallback["faithfulness"] == 0.0
    assert nr_fallback["answer_relevance"] == 0.0
    assert nr_fallback["has_hallucination"] == 0
    assert nr_fallback["judge_parse_fallback"] is True

    abs_res = normalize_judge_item({}, is_abstention=True, item_id="q-abs")
    assert abs_res["faithfulness"] == 1.0
    assert abs_res["judge_parse_fallback"] is False

    def _mock_empty_gemini(prompt: str, model_name: str):
        return []

    eval_p = tmp_path / "eval.jsonl"
    nr_p = tmp_path / "norag.jsonl"
    rag_p = tmp_path / "rag.jsonl"
    out_nr = tmp_path / "out_nr.jsonl"
    out_rag = tmp_path / "out_rag.jsonl"
    rep_p = tmp_path / "rep.md"

    eval_p.write_text(
        json.dumps({"qid_moi": "q-1", "asin": "A1", "question_text_clean": "USB?", "nhan_chot_3_muc": "yes"}) + "\n",
        encoding="utf-8",
    )
    nr_p.write_text(
        json.dumps(
            {
                "qid_moi": "q-1",
                "asin": "A1",
                "question_text_clean": "USB?",
                "nhan_chot_3_muc": "yes",
                "is_abstention": False,
                "generated_answer": "Yes.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rag_p.write_text(
        json.dumps(
            {
                "qid_moi": "q-1",
                "asin": "A1",
                "question_text_clean": "USB?",
                "nhan_chot_3_muc": "yes",
                "retrieved_contexts": ["Supports USB."],
                "is_abstention": False,
                "generated_answer": "Yes, USB.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    kq = chay_danh_gia_judge_baseline(
        eval_labeled_path=eval_p,
        no_rag_gen_path=nr_p,
        rag_gen_path=rag_p,
        out_no_rag_judge_path=out_nr,
        out_rag_judge_path=out_rag,
        report_path=rep_p,
        gemini_call_fn=_mock_empty_gemini,
    )
    assert kq["summary_no_rag"]["faithfulness_direct"] == 0.0
    assert kq["summary_rag"]["faithfulness_direct"] == 0.0
    assert kq["summary_no_rag"]["hallucination_count_total"] == 0
    assert kq["summary_rag"]["hallucination_count_total"] == 0
