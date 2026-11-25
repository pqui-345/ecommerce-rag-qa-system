"""Kiểm thử cho src/label_verification.py."""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_verification import (  # noqa: E402
    tinh_cohens_kappa,
    tong_hop_kiem_dinh_nhan,
    xac_dinh_nhan_chot,
)


def test_tinh_cohens_kappa_hoan_hao_va_ngau_nhien():
    m1 = tinh_cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0])
    assert m1["kappa"] == 1.0
    assert m1["accuracy"] == 1.0
    assert m1["f1"] == 1.0

    m0 = tinh_cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0])
    assert m0["kappa"] == 0.0
    assert m0["accuracy"] == 0.5

    me = tinh_cohens_kappa([], [])
    assert me["n"] == 0
    assert me["kappa"] == 0.0

    with pytest.raises(ValueError):
        tinh_cohens_kappa([1], [1, 0])


def test_xac_dinh_nhan_chot():
    assert xac_dinh_nhan_chot("yes", "co") == ("yes", "answerable_full", 1)
    assert xac_dinh_nhan_chot("partial", "co") == ("partial", "partial_expanded", 1)
    assert xac_dinh_nhan_chot("partial", "khong") == ("partial", "partial_10_only", 1)
    assert xac_dinh_nhan_chot("no", "co") == ("partial", "recovered_in_full_doc", 1)
    assert xac_dinh_nhan_chot("no", "khong") == ("no", "unanswerable_in_doc", 0)
    with pytest.raises(ValueError):
        xac_dinh_nhan_chot("invalid", "co")


def test_tong_hop_kiem_dinh_nhan_tren_du_lieu_gia_lap(tmp_path):
    uv_csv = tmp_path / "uv.csv"
    rs_csv = tmp_path / "rs.csv"
    q_jsonl = tmp_path / "questions_dev_clean.jsonl"
    eval_out = tmp_path / "eval.jsonl"
    report_out = tmp_path / "report.md"

    with open(uv_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "qid_moi",
                "asin",
                "question_text",
                "question_type",
                "is_answerable_auto",
                "nhan_tay",
                "ghi_chu",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "qid_moi": "q-dev-0001",
                "asin": "B01",
                "question_text": "Q1?",
                "question_type": "yesno",
                "is_answerable_auto": "1",
                "nhan_tay": "yes",
                "ghi_chu": "ok",
            }
        )
        w.writerow(
            {
                "qid_moi": "q-dev-0002",
                "asin": "B02",
                "question_text": "Q2?",
                "question_type": "descriptive",
                "is_answerable_auto": "0",
                "nhan_tay": "no",
                "ghi_chu": "none",
            }
        )

    with open(rs_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "qid_moi",
                "nhan_cuoi",
                "co_trong_tai_lieu",
                "ket_luan_muc_do",
                "bang_chung_ra_soat",
                "vi_tri_review_trong_doc",
                "ly_do_chot",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "qid_moi": "q-dev-0002",
                "nhan_cuoi": "no",
                "co_trong_tai_lieu": "khong",
                "ket_luan_muc_do": "Không có",
                "bang_chung_ra_soat": "",
                "vi_tri_review_trong_doc": "",
                "ly_do_chot": "Không tìm thấy",
            }
        )

    q_rows = [
        {
            "qid_moi": "q-dev-0001",
            "asin": "B01",
            "doc_id": "doc-B01",
            "question_text": "Q1?",
            "question_text_clean": "Q1?",
            "question_type": "yesno",
            "is_answerable_auto": 1,
            "reference_answer": "Yes.",
            "reference_helpful": [1, 1],
            "answers_clean": [],
            "review_snippets_clean": ["S1"],
            "uids": ["val-1"],
        },
        {
            "qid_moi": "q-dev-0002",
            "asin": "B02",
            "doc_id": "doc-B02",
            "question_text": "Q2?",
            "question_text_clean": "Q2?",
            "question_type": "descriptive",
            "is_answerable_auto": 0,
            "reference_answer": "No.",
            "reference_helpful": [0, 0],
            "answers_clean": [],
            "review_snippets_clean": ["S2"],
            "uids": ["val-2"],
        },
    ]
    with open(q_jsonl, "w", encoding="utf-8") as f:
        for r in q_rows:
            f.write(json.dumps(r) + "\n")

    res = tong_hop_kiem_dinh_nhan(uv_csv, rs_csv, q_jsonl, eval_out, report_out)
    assert res["n_eval"] == 2
    assert res["metrics_strict_10"]["kappa"] == 1.0
    assert eval_out.exists()
    assert report_out.exists()
