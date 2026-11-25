"""Kiểm thử cho module sinh câu trả lời Baseline src/generation_baseline.py."""

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

from src.generation_baseline import (
    chay_generation_no_rag_baseline,
    chay_generation_rag_baseline,
    la_cau_tu_choi_tra_loi,
    lam_sach_dau_ra_llm,
    tao_prompt_no_rag,
    tao_prompt_rag,
)


def _mock_llm_call(messages: Sequence[Dict[str, str]], model_name: str) -> Tuple[str, Dict]:
    user_text = messages[-1]["content"].lower()
    if "swivel" in user_text:
        ans = "I do not have enough information to determine if the stand swivels."
    else:
        ans = "<think>internal thought</think>Yes, the iPod Nano 2nd gen supports USB 2.0 syncing."
    return lam_sach_dau_ra_llm(ans), {
        "latency_sec": 0.12,
        "prompt_tokens": 50,
        "completion_tokens": 15,
        "total_tokens": 65,
    }


def test_lam_sach_dau_ra_llm_va_la_cau_tu_choi():
    raw = "<think>\nLet me check.\n</think>  The stand does not swivel.  "
    cleaned = lam_sach_dau_ra_llm(raw)
    assert cleaned == "The stand does not swivel."
    assert la_cau_tu_choi_tra_loi(cleaned) is False

    abstain_text = "I do not have enough information about this specific product."
    assert la_cau_tu_choi_tra_loi(abstain_text) is True
    assert la_cau_tu_choi_tra_loi("") is True


def test_tao_prompt_no_rag_va_rag():
    msgs_no = tao_prompt_no_rag("B001", "Electronics", "Does it have Bluetooth?")
    assert len(msgs_no) == 2
    assert msgs_no[0]["role"] == "system"
    assert "B001" in msgs_no[1]["content"]

    chunks = [{"chunk_id": "doc-B001#chunk_0000", "text": "Bluetooth 5.0 is built in."}]
    msgs_rag = tao_prompt_rag("B001", "Electronics", "Does it have Bluetooth?", chunks)
    assert len(msgs_rag) == 2
    assert "Bluetooth 5.0 is built in." in msgs_rag[1]["content"]


def test_chay_generation_no_rag_dong_hong_file_rong_va_resume(tmp_path: Path):
    empty_q = tmp_path / "empty_q.jsonl"
    empty_out = tmp_path / "empty_out.jsonl"
    empty_rep = tmp_path / "empty_rep.md"
    empty_q.write_text("", encoding="utf-8")

    kq_empty = chay_generation_no_rag_baseline(
        eval_questions_path=empty_q,
        out_jsonl_path=empty_out,
        report_path=empty_rep,
        llm_call_fn=_mock_llm_call,
    )
    assert kq_empty["n_questions"] == 0
    assert empty_rep.exists()

    eval_p = tmp_path / "eval.jsonl"
    out_p = tmp_path / "out.jsonl"
    rep_p = tmp_path / "rep.md"
    eval_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid_moi": "q-1",
                        "asin": "A1",
                        "doc_id": "doc-A1",
                        "question_text_clean": "Does the stand swivel?",
                        "question_type": "yesno",
                        "nhan_chot_3_muc": "no",
                        "nhan_chot_chi_tiet": "unanswerable_in_doc",
                        "reference_answer": "No.",
                    }
                ),
                "{dong json hong",
                json.dumps(
                    {
                        "qid_moi": "q-2",
                        "asin": "A2",
                        "doc_id": "doc-A2",
                        "question_text_clean": "Does the iPod Nano 2nd gen sync via USB?",
                        "question_type": "yesno",
                        "nhan_chot_3_muc": "yes",
                        "nhan_chot_chi_tiet": "answerable_full",
                        "reference_answer": "Yes.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kq = chay_generation_no_rag_baseline(
        eval_questions_path=eval_p,
        out_jsonl_path=out_p,
        report_path=rep_p,
        llm_call_fn=_mock_llm_call,
    )
    assert kq["n_questions"] == 2
    assert kq["n_corrupt_questions"] == 1
    assert kq["n_abstention"] == 1
    assert kq["n_direct_answer"] == 1

    call_count = {"n": 0}

    def _counting_llm(msgs, m):
        call_count["n"] += 1
        return _mock_llm_call(msgs, m)

    kq_resume = chay_generation_no_rag_baseline(
        eval_questions_path=eval_p,
        out_jsonl_path=out_p,
        report_path=rep_p,
        llm_call_fn=_counting_llm,
    )
    assert kq_resume["n_questions"] == 2
    assert call_count["n"] == 0


def test_chay_generation_rag_baseline_dong_hong_va_file_rong(tmp_path: Path):
    empty_ret = tmp_path / "empty_ret.jsonl"
    empty_out = tmp_path / "empty_rag_out.jsonl"
    empty_rep = tmp_path / "empty_rag_rep.md"
    empty_ret.write_text("", encoding="utf-8")

    kq_empty = chay_generation_rag_baseline(
        retrieval_jsonl_path=empty_ret,
        out_jsonl_path=empty_out,
        report_path=empty_rep,
        llm_call_fn=_mock_llm_call,
    )
    assert kq_empty["n_questions"] == 0

    ret_p = tmp_path / "ret.jsonl"
    out_p = tmp_path / "rag_out.jsonl"
    rep_p = tmp_path / "rag_rep.md"
    ret_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid_moi": "q-1",
                        "asin": "A2",
                        "doc_id": "doc-A2",
                        "question_text_clean": "Does the iPod Nano 2nd gen sync via USB?",
                        "nhan_chot_3_muc": "yes",
                        "nhan_chot_chi_tiet": "answerable_full",
                        "reference_answer": "Yes.",
                        "top_k_scoped": [
                            {
                                "chunk_id": "doc-A2#chunk_0000",
                                "text": "It syncs via USB 2.0 cable.",
                            }
                        ],
                    }
                ),
                "{dong json hong",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kq = chay_generation_rag_baseline(
        retrieval_jsonl_path=ret_p,
        out_jsonl_path=out_p,
        report_path=rep_p,
        llm_call_fn=_mock_llm_call,
    )
    assert kq["n_questions"] == 1
    assert kq["n_corrupt"] == 1
    assert kq["n_direct_answer"] == 1
    assert rep_p.exists()
