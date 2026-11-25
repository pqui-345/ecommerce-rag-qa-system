"""Kiem thu cho src/clean.py."""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clean import (  # noqa: E402
    chon_reference_answer,
    clean_text,
    is_dont_know_answer,
    lam_sach_questions,
)


def test_giu_nguyen_dau_sao():
    """'5*3' và dấu * giữ nguyên (quy tắc 5)."""
    assert clean_text("5*3") == "5*3"
    assert clean_text("4*4 watt") == "4*4 watt"
    assert clean_text("i like it *****") == "i like it *****"


def test_giu_nguyen_dau_nho_lon_hon():
    """'a < b > c', '<5 inch', '> 100W' giữ nguyên; các chuỗi &lt;-&gt; thật."""
    assert clean_text("a < b > c") == "a < b > c"
    assert clean_text("< 5 inch") == "< 5 inch"
    assert clean_text("> 100W") == "> 100W"
    assert clean_text("&lt;$50") == "<$50"
    assert clean_text("&lt;->S") == "<->S"
    assert clean_text("&lt;3") == "<3"
    assert clean_text("<F12>") == "<F12>"
    assert clean_text("<repeat arrow>A") == "<repeat arrow>A"
    assert clean_text("a <  b  >  c") == "a < b > c"


def test_hop_the_html_so():
    "&#34; -> \" ; &amp; -> & ; &lt;/&gt;/don&#8217;t"
    assert clean_text("&#34;hello&#34;") == '"hello"'
    assert clean_text("AT&amp;T") == "AT&T"
    assert clean_text("key&lt;10") == "key<10"
    assert clean_text("key&gt;10") == "key>10"
    assert clean_text("don&#8217;t") == "don't"
    assert clean_text("&#8220;hi&#8221;") == '"hi"'
    assert clean_text("a&#160;b") == "a b"


def test_xuong_dong_theo_quy_tac_3():
    """line1\\nline2 không còn ký tự xuống dòng theo đúng quy tắc 3."""
    assert clean_text("line1\nline2") == "line1 line2"
    assert clean_text("line1\r\nline2") == "line1 line2"
    assert clean_text("line1\r\n\r\nline2") == "line1\n\nline2"
    assert clean_text("line1\n\nline2").count("\n") == 2
    assert clean_text("A\n\n\n\nB") == "A\n\nB"


def test_rong_va_none():
    """Chuỗi rỗng/None/không phải chuỗi không làm hỏng."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""
    assert clean_text(123) == "123"


def test_xoa_the_html_co_ten():
    """<br> bị xóa nhưng '<5 inch' giữ nguyên; thẻ ngoài danh sách giữ lại."""
    assert clean_text("<br>") == ""
    assert clean_text("<br/>") == ""
    assert clean_text("line<br>line") == "lineline"
    assert clean_text("<b>bold</b>") == "bold"
    assert clean_text("trang <b> <i> i </i> </b> text") == "trang i text"
    assert clean_text("<5 inch") == "<5 inch"
    assert clean_text("<as>small") == "<as>small"
    assert clean_text("<mytag>") == "<mytag>"
    assert "<5 inch" in clean_text("This is <5 inch wide")


def test_nbsp_va_ky_tu_thong_minh():
    """NBSP / zero-width / smart quotes theo quy tắc 2."""
    assert clean_text("a\u00a0b") == "a b"
    assert clean_text("a\u200bb") == "ab"
    assert clean_text("a\u2019b") == "a'b"
    assert clean_text("1\u00a0000 VND") == "1 000 VND"


def test_gop_khoang_trang_va_strip():
    assert clean_text("a    b c ") == "a b c"
    assert clean_text("\t a  \nb \n c \t ") == "a b c"
    assert clean_text("a  b") == "a b"


def test_idempotent_cac_chuoi_kho():
    """clean_text(clean_text(x)) == clean_text(x) trên các chuỗi khó."""
    cases = [
        "5*3 &amp; 4*4",
        "&#60;$50 &amp;&amp;&#60;abc&#62;",
        "AT&T; e-mail &#34;dot net&#34;",
        "a < b > c \u00a0 \u200b<br/> d",
        "line1\nline2\r\nline3\n\n\nline4",
        "&#8217;smart&#8217; &#8220;quotes&#8221; \u2013 \u2014 \u2026",
        "H&amp;S Office H&S",
        "&amp;amp;double",
        "<5 inch > 100*3",
        "  \u00a0 leading \u200b trailing \ufeff ",
        "<b> <i> nested </i> </b> tags",
        "> 100W * 2 * < 3",
    ]
    for c in cases:
        assert clean_text(clean_text(c)) == clean_text(c), repr(c)


def _load_real_samples(n_per_file=10):
    """Lấy mẫu chuỗi thật từ docs_dev/docs_test, seed=42, từ TOÀN DỮ LIỆU."""
    samples = []
    for name in ("docs_dev.jsonl", "docs_test.jsonl"):
        path = ROOT / "data" / "processed" / name
        if not path.exists():
            continue
        with Path(path).open("r", encoding="utf-8") as f:
            lines = [l for l in (raw.strip() for raw in f) if l]
        rng = random.Random(42)
        idx = rng.sample(range(len(lines)), min(n_per_file, len(lines)))
        for i in idx:
            obj = json.loads(lines[i])
            if isinstance(obj, dict) and isinstance(obj.get("text_goc"), str):
                samples.append(obj["text_goc"])
    return samples


def test_idempotent_tren_it_nhat_20_chuoi_that():
    samples = _load_real_samples(n_per_file=10)
    assert len(samples) >= 20, "Khong đu mau that"
    for s in samples:
        once = clean_text(s)
        twice = clean_text(once)
        assert once == twice, "KHONG idempotent"


def test_chon_reference_answer_uu_tien_helpful_va_tranh_dont_know():
    answers = [
        {
            "answer_text_clean": "I don't know, sorry.",
            "helpful": [5, 5],
            "is_dont_know": True,
        },
        {
            "answer_text_clean": "Wrong answer.",
            "helpful": [0, 3],
            "is_dont_know": False,
        },
        {
            "answer_text_clean": "It uses a 2.5mm jack.",
            "helpful": [2, 2],
            "is_dont_know": False,
        },
    ]
    res = chon_reference_answer(answers)
    assert res["reference_answer"] == "It uses a 2.5mm jack."
    assert res["reference_helpful"] == [2, 2]
    assert chon_reference_answer([]) == {
        "reference_answer": "",
        "reference_helpful": [0, 0],
    }
    assert is_dont_know_answer("I have no idea about this.") is True
    assert is_dont_know_answer("It works with 12V.") is False


def test_lam_sach_questions_file_rong_va_json_hong(tmp_path):
    """Kiểm thử lam_sach_questions với file rỗng và dòng JSON hỏng."""
    empty_in = tmp_path / "empty.jsonl"
    empty_out = tmp_path / "empty_clean.jsonl"
    empty_in.write_text("", encoding="utf-8")
    r_empty = lam_sach_questions(empty_in, empty_out)
    assert r_empty["n_in"] == 0
    assert r_empty["n_out"] == 0
    assert r_empty["n_corrupt"] == 0

    mixed_in = tmp_path / "mixed.jsonl"
    mixed_out = tmp_path / "mixed_clean.jsonl"
    valid_row = {
        "qid_moi": "q-dev-0001",
        "asin": "B0001",
        "doc_id": "doc-B0001",
        "question_text": "Does it support &#34;5*3&#34;  and <10 hours?",
        "question_type": "yesno",
        "is_answerable_auto": 1,
        "answers": [
            {"answerText": "I do not know", "answerType": "NA", "helpful": [2, 2]},
            {"answerText": "Yes,  &lt;10 hours  <b>battery</b>.", "answerType": "Y", "helpful": [1, 1]},
        ],
        "uids": ["val-1"],
        "review_snippets_cau_hoi": ["Great &#34;product&#34;  for  <10 hours."],
        "dap_an_khong_biet": False,
    }
    mixed_in.write_text(
        "{corrupt json line\n" + json.dumps(valid_row) + "\n12345\n",
        encoding="utf-8",
    )
    r_mix = lam_sach_questions(mixed_in, mixed_out)
    assert r_mix["n_in"] == 1
    assert r_mix["n_out"] == 1
    assert r_mix["n_corrupt"] == 2

    out_lines = [json.loads(x) for x in mixed_out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(out_lines) == 1
    row_out = out_lines[0]
    assert row_out["question_text_clean"] == 'Does it support "5*3" and <10 hours?'
    assert row_out["reference_answer"] == "Yes, <10 hours battery."
    assert row_out["review_snippets_clean"] == ['Great "product" for <10 hours.']
