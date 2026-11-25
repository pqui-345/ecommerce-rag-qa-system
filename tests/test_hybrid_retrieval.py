"""Kiểm thử cho module truy xuất lai BM25 + Dense Vector bằng RRF src/hybrid_retrieval.py."""

import json
from pathlib import Path

import numpy as np

from src.chunking import chay_chunk_strategies, tao_chunks_tu_file_docs
from src.hybrid_retrieval import (
    chay_hybrid_6_chien_luoc,
    gop_ve_don_vi_cha,
    rrf_fuse,
    xep_hang_vector_numpy,
)


def _fake_embed(texts):
    out = []
    for t in texts:
        tl = t.lower()
        if "hdmi" in tl or "1080p" in tl:
            out.append([1.0, 0.0, 0.0])
        elif "battery" in tl or "hours" in tl:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.1, 0.1, 0.98])
    return out


def test_rrf_fuse_cong_thuc_va_hoa_diem():
    assert rrf_fuse([], k0=60) == []
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k0=60, depth=10)
    top_ids = [cid for cid, _ in fused]
    assert top_ids[:2] == ["b", "a"]
    expected_ab = round(1.0 / 61 + 1.0 / 62, 6)
    assert fused[0][1] == expected_ab
    assert fused[1][1] == expected_ab
    assert fused[2][0] == "d"


def test_xep_hang_vector_numpy_va_gop_ve_cha():
    mat = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    res_global = xep_hang_vector_numpy([1.0, 0.0], mat, allowed_indices=None, depth=2)
    assert [idx for idx, _ in res_global] == [0, 2]

    res_scoped = xep_hang_vector_numpy([1.0, 0.0], mat, allowed_indices=[1, 2], depth=2)
    assert [idx for idx, _ in res_scoped] == [2, 1]

    child_to_parent = {"c1": "p1", "c2": "p1", "c3": "p2", "c4": "p3"}
    assert gop_ve_don_vi_cha(["c1", "c2", "c3", "c4"], child_to_parent, limit=2) == ["p1", "p2"]


def test_chay_hybrid_6_chien_luoc_end_to_end(tmp_path: Path):
    docs_path = tmp_path / "docs_dev_clean.jsonl"
    doc_text = (
        "This monitor supports 1080p Full HD via HDMI port and works great.\n\n"
        "Battery life lasts up to 12 hours on a single charge for portable use."
    )
    docs_path.write_text(
        json.dumps(
            {
                "doc_id": "doc-A1",
                "asin": "A1",
                "category": "Electronics",
                "nguon_goc": "val",
                "text_clean": doc_text,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tao_chunks_tu_file_docs(
        input_path=docs_path,
        output_path=tmp_path / "chunks_dev_fixed.jsonl",
        chunk_size_words=12,
        chunk_overlap_words=2,
    )
    chay_chunk_strategies(
        processed_dir=tmp_path,
        report_path=tmp_path / "stats.md",
        strategies=("recursive", "sentence", "structure", "semantic", "parent_child"),
        params={"max_words": 14, "overlap_words": 2, "child_max_words": 7, "semantic_min_words": 5},
        embed_fn=_fake_embed,
        split="dev",
    )

    eval_q_path = tmp_path / "eval_q.jsonl"
    eval_q_path.write_text(
        json.dumps(
            {
                "qid_moi": "q-1",
                "asin": "A1",
                "doc_id": "doc-A1",
                "question_text_clean": "Does this monitor support 1080p via HDMI?",
                "nhan_chot_3_muc": "yes",
                "nhan_chot_chi_tiet": "answerable_full",
                "nhan_tay_10_snippets": "yes",
                "reference_answer": "Yes, 1080p HDMI.",
                "review_snippets_clean": [
                    "This monitor supports 1080p Full HD via HDMI port and works great."
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report_out = tmp_path / "hybrid_report.md"
    res = chay_hybrid_6_chien_luoc(
        processed_dir=tmp_path,
        eval_questions_path=eval_q_path,
        report_path=report_out,
        strategies=("fixed", "parent_child"),
        k0=60,
        depth=10,
        top_k=2,
        embed_docs_fn=_fake_embed,
        embed_queries_fn=_fake_embed,
    )
    assert report_out.exists()
    assert len(res["hybrid"]) == 2
    for hr in res["hybrid"]:
        assert hr["scoped_evidence"]["hit@5"] == 1.0
        assert 60 in hr["k0_sweep_scoped"]
