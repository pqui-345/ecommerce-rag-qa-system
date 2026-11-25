"""Kiểm thử cho module đánh giá bộ câu hỏi chuyên biệt và phản thực tế src/specialized_eval.py."""

import json
from pathlib import Path

from src.specialized_eval import (
    chay_danh_gia_counterfactual,
    chay_rag_va_judge_chinh_sach,
    danh_gia_truy_xuat_chinh_sach,
    tinh_recall_va_mrr_gold,
)


def test_tinh_recall_va_mrr_gold():
    m = tinh_recall_va_mrr_gold(["p2", "p1", "p3", "p4", "p5"], {"p1", "p3"})
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 1.0
    assert m["recall@5"] == 1.0
    assert m["mrr@5"] == 0.5

    m_empty = tinh_recall_va_mrr_gold(["p1"], set())
    assert m_empty["recall@5"] == 0.0
    assert m_empty["mrr@5"] == 0.0


def test_danh_gia_truy_xuat_chinh_sach_va_rag_va_cf(tmp_path: Path):
    docs_path = tmp_path / "policy_docs.jsonl"
    docs_path.write_text(
        json.dumps(
            {
                "doc_id": "policy-shopify-aup",
                "title": "AUP",
                "url": "https://www.shopify.com/legal/aup",
                "passages": [
                    {
                        "passage_id": "policy-shopify-aup#p000",
                        "n_words": 12,
                        "text": "Merchants must notify contract_notices@shopify.com within 7 days.",
                    },
                    {
                        "passage_id": "policy-shopify-aup#p001",
                        "n_words": 10,
                        "text": "Cookie policy explains session cookies and advertising cookies.",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    eval_p = tmp_path / "eval_policy.jsonl"
    eval_p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid_moi": "pol-1",
                        "question_category": "factoid",
                        "question_text_clean": "What email must merchants notify within 7 days?",
                        "gold_passage_ids": ["policy-shopify-aup#p000"],
                        "reference_answer": "contract_notices@shopify.com",
                    }
                ),
                json.dumps(
                    {
                        "qid_moi": "pol-2",
                        "question_category": "unanswerable",
                        "question_text_clean": "What is the fee in VND?",
                        "gold_passage_ids": [],
                        "reference_answer": "",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_emb(texts):
        return [[1.0, 0.0] if "7 days" in t or "email" in t.lower() else [0.0, 1.0] for t in texts]

    ret_out = tmp_path / "ret_pol.jsonl"
    ret_rep = tmp_path / "ret_rep.md"
    res_ret = danh_gia_truy_xuat_chinh_sach(
        policy_docs_path=docs_path,
        eval_policy_path=eval_p,
        out_hybrid_jsonl_path=ret_out,
        report_path=ret_rep,
        embed_docs_fn=fake_emb,
        embed_queries_fn=fake_emb,
        top_k=2,
    )
    assert res_ret["summary"]["hybrid_rrf"]["recall@1"] == 1.0
    assert ret_rep.exists()

    def fake_llm(msgs, m):
        u = msgs[-1]["content"].lower()
        if "vnd" in u:
            return "The provided policy documents do not contain enough information.", {"total_tokens": 10}
        return "Notify contract_notices@shopify.com within 7 days.", {"total_tokens": 10}

    def fake_gem_pol(prompt, m):
        return [
            {"qid_moi": "pol-1", "chunk_relevance": [1, 0, 0, 0, 0], "faithfulness": 1.0, "answer_relevance": 1.0, "has_hallucination": 0, "handles_ambiguity": 0},
            {"qid_moi": "pol-2", "chunk_relevance": [0, 0, 0, 0, 0], "faithfulness": 1.0, "answer_relevance": 0.0, "has_hallucination": 0, "handles_ambiguity": 0},
        ]

    res_pol = chay_rag_va_judge_chinh_sach(
        retrieval_policy_path=ret_out,
        out_gen_path=tmp_path / "gen_pol.jsonl",
        out_judge_path=tmp_path / "judge_pol.jsonl",
        report_path=tmp_path / "pol_rag_rep.md",
        llm_call_fn=fake_llm,
        gemini_call_fn=fake_gem_pol,
    )
    assert res_pol["n_total"] == 2

    cf_path = tmp_path / "cf.jsonl"
    cf_path.write_text(
        json.dumps(
            {
                "qid_moi": "q-1",
                "asin": "A1",
                "question_text_clean": "Is it 110V or 220V?",
                "original_fact": "Universal 100V-240V",
                "counterfactual_fact": "Strictly 110V only",
                "counterfactual_contexts": ["Strictly 110V only."],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_gem_cf(prompt, m):
        return [{"qid_moi": "q-1", "adheres_counterfactual": 1, "reverts_to_prior": 0}]

    res_cf = chay_danh_gia_counterfactual(
        cf_path=cf_path,
        out_gen_path=tmp_path / "gen_cf.jsonl",
        out_judge_path=tmp_path / "judge_cf.jsonl",
        report_path=tmp_path / "cf_rep.md",
        llm_call_fn=lambda msgs, m: ("It is strictly 110V only.", {"total_tokens": 10}),
        gemini_call_fn=fake_gem_cf,
    )
    assert res_cf["adherence_rate"] == 100.0
