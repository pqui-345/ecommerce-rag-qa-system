"""Bộ kiểm thử đơn vị cho chức năng dựng bộ làm việc nhỏ (working_set)."""

import json
from pathlib import Path

import pytest

from src.working_set import bo_lam_viec, is_dont_know, norm_text, segment_key, stats_of


def _make_merged(tmp_path: Path, records):
    p = tmp_path / "merged_v2.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def _base_record(qid, asin, source, category, question_text, answers=None, snippets=None, is_answerable=1, question_type="descriptive"):
    return {
        "qid": qid,
        "asin": asin,
        "category": category,
        "nguon_goc": source,
        "questionText": question_text,
        "questionType": question_type,
        "is_answerable": is_answerable,
        "answers": answers or [{"answerText": "ans " + str(qid), "answerType": "NA", "helpful": [1, 1]}],
        "review_snippets": snippets or [f"snippet {qid}"],
        "uid": f"{source}-{qid}",
    }


def test_gop_doan_loai_trung_dung_thu_tu(tmp_path: Path):
    """Gộp đoạn review_snippets loại trùng đúng thứ tự gặp đầu tiên."""
    merged = _make_merged(tmp_path, [
        _base_record(1, "A1", "val", "Electronics", "Q1?", snippets=["abc def", "abc def", "ghi"]),
    ])
    out = tmp_path / "out"
    report = tmp_path / "report"
    bo_lam_viec(
        merged_path=merged,
        processed_dir=out,
        report_dir=report,
        log_path=tmp_path / "log.txt",
        category="Electronics",
        min_doc_chars=0,
        n_asin_dev=1,
        n_asin_test=0,
        seed=42,
        min_question_chars=0,
        n_kiem_tay=0,
    )
    docs = [json.loads(l) for l in (out / "docs_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(docs) == 1
    assert docs[0]["snippets"] == ["abc def", "ghi"], docs[0]["snippets"]
    assert docs[0]["text_goc"] == "abc def\n\nghi"


def test_gop_cau_hoi_trung_giU_dung_answers(tmp_path: Path):
    """Cùng (asin, questionText_chuẩn) => gộp thành một câu hỏi, giữ đủ answers."""
    merged = _make_merged(tmp_path, [
        _base_record(1, "A1", "val", "Electronics", "Hello World?", answers=[{"answerText": "a1"}]),
        _base_record(2, "A1", "val", "Electronics", "hello   world?", answers=[{"answerText": "a2"}, {"answerText": "a3"}]),
    ])
    out = tmp_path / "out"
    report = tmp_path / "report"
    bo_lam_viec(
        merged_path=merged,
        processed_dir=out,
        report_dir=report,
        log_path=tmp_path / "log.txt",
        category="Electronics",
        min_doc_chars=0,
        n_asin_dev=1,
        n_asin_test=0,
        seed=42,
        min_question_chars=0,
        n_kiem_tay=0,
    )
    qs = [json.loads(l) for l in (out / "questions_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(qs) == 1, qs
    assert len(qs[0]["answers"]) == 3, qs[0]["answers"]
    ans_texts = sorted(a["answerText"] for a in qs[0]["answers"])
    assert ans_texts == ["a1", "a2", "a3"], ans_texts
    assert set(qs[0]["uids"]) == {"val-1", "val-2"}, qs[0]["uids"]


def test_chon_mau_tai_lap_duoc_vung_cung_seed(tmp_path: Path):
    """Chạy hai lần với cùng SEED => kết quả asin/dev/test giống hệt nhau."""
    recs = []
    for i in range(50):
        recs.append(_base_record(i, f"A{i%10}", "val", "Electronics", f"Q{i}?", snippets=[f"s{i}"] * 20))
        recs.append(_base_record(i + 100, f"B{i%10}", "test", "Electronics", f"T{i}?", snippets=[f"t{i}"] * 20))
    merged = _make_merged(tmp_path, recs)

    def run_once(tag):
        out = tmp_path / f"out_{tag}"
        report = tmp_path / f"report_{tag}"
        bo_lam_viec(
            merged_path=merged,
            processed_dir=out,
            report_dir=report,
            log_path=tmp_path / f"log_{tag}.txt",
            category="Electronics",
            min_doc_chars=0,
            n_asin_dev=3,
            n_asin_test=3,
            seed=42,
            min_question_chars=0,
            n_kiem_tay=0,
        )
        dev_lines = (out / "docs_dev.jsonl").read_text(encoding="utf-8").splitlines()
        test_lines = (out / "docs_test.jsonl").read_text(encoding="utf-8").splitlines()
        dev = sorted(json.loads(d)["asin"] for d in dev_lines if d.strip())
        test = sorted(json.loads(d)["asin"] for d in test_lines if d.strip())
        return dev, test

    dev1, test1 = run_once("a")
    dev2, test2 = run_once("b")
    assert dev1 == dev2
    assert test1 == test2


def test_kiem_tra_ro_ri_asin_giua_dev_test(tmp_path: Path):
    """Không có asin nào nằm đồng thời ở dev và test."""
    recs = []
    for i in range(30):
        recs.append(_base_record(i, f"D{i}", "val", "Electronics", f"V{i}?", snippets=["x"] * 30))
        recs.append(_base_record(i, f"T{i}", "test", "Electronics", f"Te{i}?", snippets=["y"] * 30))
    merged = _make_merged(tmp_path, recs)
    out = tmp_path / "out"
    report = tmp_path / "report"
    bo_lam_viec(
        merged_path=merged,
        processed_dir=out,
        report_dir=report,
        log_path=tmp_path / "log.txt",
        category="Electronics",
        min_doc_chars=0,
        n_asin_dev=5,
        n_asin_test=5,
        seed=42,
        min_question_chars=0,
        n_kiem_tay=0,
    )
    dev_lines = (out / "docs_dev.jsonl").read_text(encoding="utf-8").splitlines()
    test_lines = (out / "docs_test.jsonl").read_text(encoding="utf-8").splitlines()
    dev_asin = set(json.loads(d)["asin"] for d in dev_lines if d.strip())
    test_asin = set(json.loads(d)["asin"] for d in test_lines if d.strip())
    assert dev_asin & test_asin == set()


def test_regex_dont_know_bien_the():
    """Regex 'don't know' chấp nhận các biến thể."""
    assert is_dont_know("I don't know") is True
    assert is_dont_know("I dont know") is True
    assert is_dont_know("I do not know") is True
    assert is_dont_know("idk") is True
    assert is_dont_know("IDK") is True
    assert is_dont_know("not sure") is True
    assert is_dont_know("No idea") is True
    assert is_dont_know("I don’t know") is True
    assert is_dont_know("I know the answer") is False
    assert is_dont_know("") is False


def test_dong_thieu_truong_khong_lam_sap(tmp_path: Path):
    """Dòng thiếu trường không làm sập chương trình."""
    merged = _make_merged(tmp_path, [
        {"qid": 1, "asin": "A1", "category": "Electronics", "nguon_goc": "val"},
        _base_record(2, "A1", "val", "Electronics", "Q2?", snippets=["s2"] * 30),
    ])
    out = tmp_path / "out"
    report = tmp_path / "report"
    bo_lam_viec(
        merged_path=merged,
        processed_dir=out,
        report_dir=report,
        log_path=tmp_path / "log.txt",
        category="Electronics",
        min_doc_chars=0,
        n_asin_dev=1,
        n_asin_test=0,
        seed=42,
        min_question_chars=0,
        n_kiem_tay=0,
    )
    assert (out / "docs_dev.jsonl").exists()


def test_file_rong_khong_loi(tmp_path: Path):
    """File merged rỗng không gây lỗi."""
    merged = _make_merged(tmp_path, [])
    out = tmp_path / "out"
    report = tmp_path / "report"
    bo_lam_viec(
        merged_path=merged,
        processed_dir=out,
        report_dir=report,
        log_path=tmp_path / "log.txt",
        category="Electronics",
        min_doc_chars=0,
        n_asin_dev=0,
        n_asin_test=0,
        seed=42,
        min_question_chars=0,
        n_kiem_tay=0,
    )
    assert (out / "docs_dev.jsonl").exists()
    assert (out / "docs_test.jsonl").exists()
    assert (out / "questions_dev.jsonl").exists()
    assert (out / "questions_test.jsonl").exists()


def test_norm_text_va_segment_key():
    assert norm_text("  Hello   World  ") == "hello world"
    assert segment_key("Abc Def") == segment_key("abc   def")
    assert segment_key("x") != segment_key("y")


def test_stats_of():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mn, med, p90, mx = stats_of(vals)
    assert mn == 1
    assert mx == 10
    assert med == 5.5
