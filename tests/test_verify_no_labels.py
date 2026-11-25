"""Kiem thu cho src/verify_no_labels.py: ham tach doan, loai trung, BM25, stopword."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.verify_no_labels import (  # noqa: E402
    BM25TinhDiem, bo_stopword, doan_cau_hoi_da_lam_sach, loai_doan_goc,
    tach_doan, tokenize,
)


def test_tach_doan_co_ban():
    assert tach_doan("A\n\nB") == ["A", "B"]
    assert tach_doan("ABC") == ["ABC"]
    assert tach_doan("A\n\n\n\nB") == ["A", "B"]
    assert tach_doan("\n\nA") == ["A"]
    assert tach_doan("A\n\n") == ["A"]


def test_tach_doan_bo_doan_rong():
    assert tach_doan("   \n\n   ") == []
    assert tach_doan("A\n\n   \n\nB") == ["A", "B"]
    assert tach_doan("A\n\n\n\n\n\nB") == ["A", "B"]


def test_tach_doan_an_toan_dau_vao_le():
    assert tach_doan("") == []
    assert tach_doan(None) == []
    assert tach_doan(123) == []


def test_tach_doan_giu_nguyen_nguyen_doan():
    """Đoạn giữ nguyên nội dung (chỉ tách, không strip bên trong)."""
    assert tach_doan("  A B  \n\nC D") == ["  A B  ", "C D"]



def test_loai_doan_goc_theo_chuan_hoa():
    doc = ["The battery is good.", "Screen is bright."]
    cau = ["the  battery   IS GOOD."]
    assert loai_doan_goc(doc, cau) == ["Screen is bright."]
    assert doc == ["The battery is good.", "Screen is bright."]


def test_loai_doan_goc_giu_thu_tu():
    doc = ["s1", "s2", "s3"]
    cau = ["S2"]
    assert loai_doan_goc(doc, cau) == ["s1", "s3"]


def test_loai_doan_goc_khong_trung_ka():
    assert loai_doan_goc(["x", "y"], ["z"]) == ["x", "y"]
    assert loai_doan_goc(["x"], []) == ["x"]
    assert loai_doan_goc([], ["x"]) == []
    assert loai_doan_goc(None, None) == []


def test_doan_cau_hoi_da_lam_sach_giai_ma_entity():
    """Đoạn gốc chứa &#34; phải được giải mã để khớp đoạn đã làm sạch."""
    goc = 'partition for a 15&#34; laptop'
    da_sach = doan_cau_hoi_da_lam_sach([goc])
    assert da_sach == ['partition for a 15" laptop']
    doc = ['partition for a 15" laptop', "khác"]
    assert loai_doan_goc(doc, da_sach) == ["khác"]


def test_doan_cau_hoi_da_lam_sach_an_toan():
    assert doan_cau_hoi_da_lam_sach([]) == []
    assert doan_cau_hoi_da_lam_sach(None) == []
    assert doan_cau_hoi_da_lam_sach(["  A  B  "]) == ["A B"]
    assert doan_cau_hoi_da_lam_sach([42]) == []



def test_tokenize_va_bo_stopword():
    assert bo_stopword(tokenize("What is the best way to charge it?")) == [
        "best", "way", "charge",
    ]
    assert bo_stopword(tokenize("Does it support 5 GHz version 802.11?")) == [
        "support", "5", "ghz", "version", "802", "11",
    ]
    assert bo_stopword(tokenize("")) == []
    assert bo_stopword(None) == []
    assert bo_stopword(tokenize("I don't know")) == ["know"]


def test_tokenize_giu_so():
    assert "5" in tokenize("Top 5 picks")
    assert "2020" in tokenize("since 2020")



def test_bm25_diem_tu_khop_cao_hon_khong_khop():
    corpus = [
        tokenize("the battery lasts long battery battery"),
        tokenize("the screen is bright and clear"),
    ]
    bm = BM25TinhDiem(corpus)
    q = ["battery"]
    assert bm.diem(q, 0) > 0.0
    assert bm.diem(q, 1) == 0.0


def test_bm25_tu_khong_co_trong_corpus():
    bm = BM25TinhDiem([tokenize("abc def")])
    assert bm.diem(["zzz"], 0) == 0.0


def test_bm25_dau_vao_rong():
    bm = BM25TinhDiem([])
    assert bm.diem(["x"], 0) == 0.0
    bm2 = BM25TinhDiem([tokenize("word")])
    assert bm2.diem([], 0) == 0.0


def test_bm25_tf_cao_diem_cao():
    corpus = [tokenize("battery " * 10), tokenize("battery")]
    bm = BM25TinhDiem(corpus)
    assert bm.diem(["battery"], 0) > bm.diem(["battery"], 1)
