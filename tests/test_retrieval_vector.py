"""Kiểm thử cho module truy xuất Dense Vector src/retrieval_vector.py."""

import json
import math
from pathlib import Path
from typing import List, Sequence

from src.retrieval_vector import (
    chay_danh_gia_vector_baseline,
    tao_hoac_nap_chroma_collection,
    truy_xuat_chroma_top_k,
)


def _dummy_embed(texts: Sequence[str]) -> List[List[float]]:
    """Hàm nhúng giả lập 4 chiều tất định để kiểm thử nhanh ChromaDB không cần tải model."""
    vecs: List[List[float]] = []
    for t in texts:
        tl = (t or "").lower()
        v = [
            1.0 if "swivel" in tl or "stand" in tl else 0.1,
            1.0 if "battery" in tl or "hours" in tl else 0.1,
            1.0 if "bluetooth" in tl or "sony" in tl else 0.1,
            0.2,
        ]
        norm = math.sqrt(sum(x * x for x in v))
        vecs.append([x / norm for x in v])
    return vecs


def test_tao_collection_va_truy_xuat_chroma_top_k():
    chunks = [
        {
            "chunk_id": "doc-1#chunk_0000",
            "doc_id": "doc-1",
            "asin": "A1",
            "category": "Electronics",
            "chunk_index": 0,
            "n_words": 9,
            "text": "The stand is fixed and does not swivel.",
        },
        {
            "chunk_id": "doc-2#chunk_0000",
            "doc_id": "doc-2",
            "asin": "A2",
            "category": "Electronics",
            "chunk_index": 0,
            "n_words": 8,
            "text": "Battery life is 20 hours on full charge.",
        },
    ]
    col = tao_hoac_nap_chroma_collection(
        chunks=chunks,
        embed_docs_fn=_dummy_embed,
        collection_name="test_col_unit",
        persist_dir=None,
        force_rebuild=True,
    )
    q_vec = _dummy_embed(["Does the stand swivel?"])[0]
    top_global = truy_xuat_chroma_top_k(col, q_vec, top_k=2)
    assert len(top_global) == 2
    assert top_global[0]["chunk_id"] == "doc-1#chunk_0000"
    assert top_global[0]["score"] > top_global[1]["score"]

    top_scoped = truy_xuat_chroma_top_k(
        col, q_vec, top_k=1, where_filter={"doc_id": "doc-2"}
    )
    assert len(top_scoped) == 1
    assert top_scoped[0]["chunk_id"] == "doc-2#chunk_0000"


def test_chay_danh_gia_vector_baseline_dong_hong_va_file_rong(tmp_path: Path):
    empty_c = tmp_path / "empty_c.jsonl"
    empty_q = tmp_path / "empty_q.jsonl"
    empty_out = tmp_path / "empty_out.jsonl"
    empty_rep = tmp_path / "empty_rep.md"
    empty_c.write_text("", encoding="utf-8")
    empty_q.write_text("", encoding="utf-8")

    kq_empty = chay_danh_gia_vector_baseline(
        chunks_path=empty_c,
        eval_questions_path=empty_q,
        out_jsonl_path=empty_out,
        report_path=empty_rep,
        persist_dir=None,
        embed_docs_fn=_dummy_embed,
        embed_queries_fn=_dummy_embed,
    )
    assert kq_empty["n_chunks"] == 0
    assert kq_empty["n_questions"] == 0

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
                        "category": "Electronics",
                        "chunk_index": 0,
                        "n_words": 9,
                        "text": "The stand is fixed and does not swivel.",
                    }
                ),
                "{dong json hong",
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
                        "review_snippets_clean": ["The stand is fixed and does not swivel."],
                        "reference_answer": "No.",
                    }
                ),
                "{json cau hoi hong",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kq = chay_danh_gia_vector_baseline(
        chunks_path=chunks_p,
        eval_questions_path=eval_p,
        out_jsonl_path=out_p,
        report_path=rep_p,
        persist_dir=None,
        embed_docs_fn=_dummy_embed,
        embed_queries_fn=_dummy_embed,
    )
    assert kq["n_chunks"] == 1
    assert kq["n_questions"] == 1
    assert kq["n_corrupt_chunks"] == 1
    assert kq["n_corrupt_questions"] == 1
    assert kq["scoped_evidence"]["hit@1"] == 1.0
    assert rep_p.exists()
