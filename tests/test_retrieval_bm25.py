"""Kiểm thử cho module truy xuất BM25 cơ sở src/retrieval_bm25.py."""

import json
from pathlib import Path

from src.verify_no_labels import BM25TinhDiem, bo_stopword, tokenize
from src.retrieval_bm25 import (
    chay_danh_gia_bm25_baseline,
    doc_danh_sach_jsonl,
    tinh_hit_va_mrr,
    truy_xuat_bm25_top_k,
    xac_dinh_ground_truth_chunks,
)


def test_tinh_hit_va_mrr():
    ranked = [
        {"rank": 1, "chunk_id": "c1", "doc_id": "d1"},
        {"rank": 2, "chunk_id": "c2", "doc_id": "d2"},
        {"rank": 3, "chunk_id": "c3", "doc_id": "d3"},
        {"rank": 4, "chunk_id": "c4", "doc_id": "d4"},
        {"rank": 5, "chunk_id": "c5", "doc_id": "d5"},
    ]
    m1 = tinh_hit_va_mrr(ranked, "chunk_id", {"c2"})
    assert m1["hit@1"] == 0.0
    assert m1["hit@3"] == 1.0
    assert m1["hit@5"] == 1.0
    assert m1["mrr@5"] == 0.5

    m_miss = tinh_hit_va_mrr(ranked, "chunk_id", {"c99"})
    assert m_miss["hit@1"] == 0.0
    assert m_miss["hit@5"] == 0.0
    assert m_miss["mrr@5"] == 0.0


def test_xac_dinh_ground_truth_chunks_yes_va_no():
    doc_chunks = [
        {
            "chunk_id": "doc-1#chunk_0000",
            "doc_id": "doc-1",
            "text": "The battery life of this headset is around 15 hours on a single charge.",
        },
        {
            "chunk_id": "doc-1#chunk_0001",
            "doc_id": "doc-1",
            "text": "Bluetooth pairing is fast and supports multipoint connection with two phones.",
        },
    ]
    q_yes = {
        "nhan_chot_3_muc": "yes",
        "nhan_tay_10_snippets": "yes",
        "review_snippets_clean": [
            "The battery life of this headset is around 15 hours on a single charge."
        ],
        "bang_chung_ra_soat": "",
    }
    gt_yes = xac_dinh_ground_truth_chunks(q_yes, doc_chunks)
    assert gt_yes == {"doc-1#chunk_0000"}

    q_no = {
        "nhan_chot_3_muc": "no",
        "nhan_tay_10_snippets": "no",
        "review_snippets_clean": [
            "The battery life of this headset is around 15 hours on a single charge."
        ],
        "bang_chung_ra_soat": "Đã rà 10 review còn lại; không thấy bằng chứng.",
    }
    gt_no = xac_dinh_ground_truth_chunks(q_no, doc_chunks)
    assert gt_no == set()


def test_truy_xuat_bm25_top_k_global_va_scoped():
    chunks = [
        {
            "chunk_id": "doc-1#chunk_0000",
            "doc_id": "doc-1",
            "asin": "A1",
            "chunk_index": 0,
            "n_words": 8,
            "text": "Actiontec modem works with CenturyLink VDSL2 service",
        },
        {
            "chunk_id": "doc-2#chunk_0000",
            "doc_id": "doc-2",
            "asin": "A2",
            "chunk_index": 0,
            "n_words": 8,
            "text": "Sony headphones have great bass and noise cancellation",
        },
    ]
    corpus = [bo_stopword(tokenize(c["text"])) for c in chunks]
    bm25 = BM25TinhDiem(corpus)

    top_global = truy_xuat_bm25_top_k(bm25, chunks, "Does it work with CenturyLink VDSL2?", top_k=2)
    assert top_global[0]["chunk_id"] == "doc-1#chunk_0000"

    top_scoped = truy_xuat_bm25_top_k(
        bm25, chunks, "Does it work with CenturyLink VDSL2?", top_k=2, allowed_indices=[1]
    )
    assert len(top_scoped) == 1
    assert top_scoped[0]["chunk_id"] == "doc-2#chunk_0000"


def test_doc_danh_sach_jsonl_va_chay_pipeline_dong_hong_file_rong(tmp_path: Path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    items, n_err = doc_danh_sach_jsonl(empty_file)
    assert items == []
    assert n_err == 0

    chunks_p = tmp_path / "chunks.jsonl"
    eval_p = tmp_path / "eval.jsonl"
    out_p = tmp_path / "out.jsonl"
    rep_p = tmp_path / "report.md"

    chunks_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "doc-A1#chunk_0000",
                        "doc_id": "doc-A1",
                        "asin": "A1",
                        "chunk_index": 0,
                        "n_words": 10,
                        "text": "The stand is fixed and does not swivel at all.",
                    }
                ),
                "{json hong",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    eval_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid_moi": "q-1",
                        "asin": "A1",
                        "doc_id": "doc-A1",
                        "question_text_clean": "Does the stand swivel?",
                        "nhan_chot_3_muc": "yes",
                        "nhan_chot_chi_tiet": "answerable_full",
                        "nhan_tay_10_snippets": "yes",
                        "review_snippets_clean": ["The stand is fixed and does not swivel at all."],
                        "reference_answer": "No, it does not swivel.",
                    }
                ),
                "{json cau hoi hong",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kq = chay_danh_gia_bm25_baseline(chunks_p, eval_p, out_p, rep_p, top_k=5)
    assert kq["n_chunks"] == 1
    assert kq["n_questions"] == 1
    assert kq["n_corrupt_chunks"] == 1
    assert kq["n_corrupt_questions"] == 1
    assert kq["scoped_evidence"]["hit@1"] == 1.0
    assert rep_p.exists()
