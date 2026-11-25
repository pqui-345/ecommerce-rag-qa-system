"""Kiểm thử cho module đối sánh truy xuất 6 chiến lược phân đoạn src/chunking_eval.py."""

import json
from pathlib import Path

from src.chunking import chay_chunk_strategies, tao_chunks_tu_file_docs
from src.chunking_eval import (
    _dedup_children_to_parents,
    bootstrap_diff_ci,
    chay_danh_gia_truy_xuat_6_chien_luoc,
    mcnemar_exact_pvalue,
)


def _fake_embed_docs(texts):
    out = []
    for t in texts:
        tl = t.lower()
        if "1080p" in tl or "hdmi" in tl:
            out.append([1.0, 0.0, 0.0])
        elif "battery" in tl or "hours" in tl:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.2, 0.2, 0.9])
    return out


def _fake_embed_queries(texts):
    return _fake_embed_docs(texts)


def test_mcnemar_va_bootstrap():
    mc_same = mcnemar_exact_pvalue([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert mc_same["b_only_base"] == 0
    assert mc_same["c_only_new"] == 0
    assert mc_same["p_value"] == 1.0

    mc_diff = mcnemar_exact_pvalue([0.0] * 10, [1.0] * 10)
    assert mc_diff["b_only_base"] == 0
    assert mc_diff["c_only_new"] == 10
    assert mc_diff["p_value"] < 0.01

    bt = bootstrap_diff_ci([0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], n_boot=200, seed=42)
    assert bt["diff_mean"] == 0.5
    assert bt["ci_low"] == 0.5
    assert bt["ci_high"] == 0.5


def test_dedup_children_to_parents():
    parents = {
        "doc-1#pc_parent_0000": {
            "chunk_id": "doc-1#pc_parent_0000",
            "doc_id": "doc-1",
            "asin": "A1",
            "chunk_index": 0,
            "n_words": 50,
            "text": "parent zero text",
        },
        "doc-1#pc_parent_0001": {
            "chunk_id": "doc-1#pc_parent_0001",
            "doc_id": "doc-1",
            "asin": "A1",
            "chunk_index": 1,
            "n_words": 60,
            "text": "parent one text",
        },
    }
    children_ranked = [
        {"rank": 1, "score": 0.95, "chunk_id": "c0_0", "parent_id": "doc-1#pc_parent_0000"},
        {"rank": 2, "score": 0.90, "chunk_id": "c0_1", "parent_id": "doc-1#pc_parent_0000"},
        {"rank": 3, "score": 0.85, "chunk_id": "c1_0", "parent_id": "doc-1#pc_parent_0001"},
    ]
    deduped = _dedup_children_to_parents(children_ranked, parents, top_k=2)
    assert len(deduped) == 2
    assert deduped[0]["chunk_id"] == "doc-1#pc_parent_0000"
    assert deduped[0]["rank"] == 1
    assert deduped[1]["chunk_id"] == "doc-1#pc_parent_0001"
    assert deduped[1]["rank"] == 2


def test_chay_danh_gia_truy_xuat_6_chien_luoc_end_to_end(tmp_path: Path):
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
        report_path=tmp_path / "chunk_stats.md",
        strategies=("recursive", "sentence", "structure", "semantic", "parent_child"),
        params={"max_words": 14, "overlap_words": 2, "child_max_words": 7, "semantic_min_words": 5},
        embed_fn=_fake_embed_docs,
        split="dev",
    )

    eval_q_path = tmp_path / "eval_120.jsonl"
    questions = [
        {
            "qid_moi": "q-1",
            "asin": "A1",
            "doc_id": "doc-A1",
            "question_text_clean": "Does this monitor support 1080p via HDMI?",
            "nhan_chot_3_muc": "yes",
            "nhan_chot_chi_tiet": "answerable_full",
            "nhan_tay_10_snippets": "yes",
            "reference_answer": "Yes, 1080p via HDMI.",
            "review_snippets_clean": [
                "This monitor supports 1080p Full HD via HDMI port and works great."
            ],
        },
        {
            "qid_moi": "q-2",
            "asin": "A1",
            "doc_id": "doc-A1",
            "question_text_clean": "Is it waterproof under 50 meters?",
            "nhan_chot_3_muc": "no",
            "nhan_chot_chi_tiet": "unanswerable_in_doc",
            "nhan_tay_10_snippets": "no",
            "reference_answer": "",
            "review_snippets_clean": [],
        },
    ]
    eval_q_path.write_text(
        "\n".join(json.dumps(q) for q in questions) + "\n",
        encoding="utf-8",
    )

    report_out = tmp_path / "retrieval_comp.md"
    res = chay_danh_gia_truy_xuat_6_chien_luoc(
        processed_dir=tmp_path,
        eval_questions_path=eval_q_path,
        report_path=report_out,
        embed_docs_fn=_fake_embed_docs,
        embed_queries_fn=_fake_embed_queries,
        top_k=2,
        reuse_existing=False,
    )
    assert report_out.exists()
    assert len(res["bm25"]) == 6
    assert len(res["vector"]) == 6
    for r in res["vector"]:
        assert r["scoped_evidence"]["hit@5"] == 1.0
