"""Bộ kiểm thử đơn vị cho chức năng gộp dữ liệu."""

import json
from pathlib import Path
import pytest

from src.merge import gop_du_lieu


def test_gop_dung_thu_tu_va_nguon_goc(tmp_path: Path):
    """Kiểm tra gộp đúng thứ tự (val trước, test sau) và gắn đúng nhãn nguon_goc."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    val_data = [{"qid": 1, "text": "val_1"}, {"qid": 2, "text": "val_2"}]
    test_data = [{"qid": 3, "text": "test_1"}, {"qid": 4, "text": "test_2"}]

    val_file.write_text("\n".join(json.dumps(d) for d in val_data) + "\n", encoding="utf-8")
    test_file.write_text("\n".join(json.dumps(d) for d in test_data) + "\n", encoding="utf-8")

    stats = gop_du_lieu(val_file, test_file, out_file, dedup_key="qid", log_path=log_file)

    assert stats["total_written"] == 4
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 4

    assert lines[0]["nguon_goc"] == "val" and lines[0]["qid"] == 1
    assert lines[1]["nguon_goc"] == "val" and lines[1]["qid"] == 2
    assert lines[2]["nguon_goc"] == "test" and lines[2]["qid"] == 3
    assert lines[3]["nguon_goc"] == "test" and lines[3]["qid"] == 4


def test_qid_trung_chi_giu_mot_dong(tmp_path: Path):
    """Kiểm tra qid trùng nhau chỉ giữ dòng xuất hiện đầu tiên."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    val_data = [{"qid": 10, "text": "first_occurrence_val"}, {"qid": 20, "text": "v2"}]
    test_data = [
        {"qid": 10, "text": "second_occurrence_test"},
        {"qid": 30, "text": "t3"},
        {"qid": 30, "text": "t3_duplicate"},
    ]

    val_file.write_text("\n".join(json.dumps(d) for d in val_data) + "\n", encoding="utf-8")
    test_file.write_text("\n".join(json.dumps(d) for d in test_data) + "\n", encoding="utf-8")

    stats = gop_du_lieu(val_file, test_file, out_file, dedup_key="qid", log_path=log_file)

    assert stats["dedup_dropped"] == 2
    assert stats["total_written"] == 3

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    qids = [d["qid"] for d in lines]
    assert qids == [10, 20, 30]

    assert lines[0]["qid"] == 10
    assert lines[0]["nguon_goc"] == "val"
    assert lines[0]["text"] == "first_occurrence_val"


def test_dong_thieu_qid_van_duoc_ghi(tmp_path: Path):
    """Kiểm tra dòng thiếu qid không bị loại, vẫn được ghi ra và đếm riêng."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    val_data = [{"qid": 1, "text": "co_qid"}, {"text": "thieu_qid_val"}]
    test_data = [{"text": "thieu_qid_test"}, {"qid": 2, "text": "co_qid_2"}]

    val_file.write_text("\n".join(json.dumps(d) for d in val_data) + "\n", encoding="utf-8")
    test_file.write_text("\n".join(json.dumps(d) for d in test_data) + "\n", encoding="utf-8")

    stats = gop_du_lieu(val_file, test_file, out_file, dedup_key="qid", log_path=log_file)

    assert stats["missing_key"] == 2
    assert stats["total_written"] == 4

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    missing_records = [d for d in lines if "qid" not in d]
    assert len(missing_records) == 2
    assert missing_records[0]["text"] == "thieu_qid_val" and missing_records[0]["nguon_goc"] == "val"
    assert missing_records[1]["text"] == "thieu_qid_test" and missing_records[1]["nguon_goc"] == "test"


def test_dong_json_hong_khong_lam_sap_va_duoc_dem(tmp_path: Path):
    """Kiểm tra dòng JSON hỏng bị bỏ qua, đếm đúng và không làm sập chương trình."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    val_content = '{"qid": 1, "text": "valid_1"}\n{broken json line\n{"qid": 2, "text": "valid_2"}\n'
    test_content = 'not json at all!\n{"qid": 3, "text": "valid_3"}\n{"unclosed": "brace"\n'

    val_file.write_text(val_content, encoding="utf-8")
    test_file.write_text(test_content, encoding="utf-8")

    stats = gop_du_lieu(val_file, test_file, out_file, dedup_key="qid", log_path=log_file)

    assert stats["val"]["corrupted"] == 1
    assert stats["test"]["corrupted"] == 2
    assert stats["total_written"] == 3

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 3
    assert [d["qid"] for d in lines] == [1, 2, 3]


def test_file_rong_khong_loi(tmp_path: Path):
    """Kiểm tra trường hợp file rỗng chạy trơn tru không phát sinh lỗi."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    val_file.write_text("", encoding="utf-8")
    test_file.write_text("", encoding="utf-8")

    stats = gop_du_lieu(val_file, test_file, out_file, dedup_key="qid", log_path=log_file)

    assert stats["val"]["read"] == 0
    assert stats["test"]["read"] == 0
    assert stats["total_written"] == 0
    assert stats["out_size_bytes"] == 0
    assert out_file.exists()


def test_cac_truong_goc_khong_bi_thay_doi(tmp_path: Path):
    """Kiểm tra các trường gốc được giữ nguyên vẹn 100%, chỉ thêm trường nguon_goc."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    original_record = {
        "asin": "B009B0MZ8U",
        "category": "Electronics",
        "questionText": "What ISO range is covered?",
        "questionType": "descriptive",
        "review_snippets": ["Great camera", "Low light performance is solid"],
        "answers": [{"answerText": "100 to 6400", "helpful": [5, 5]}],
        "is_answerable": 1,
        "qid": 9999,
    }

    val_file.write_text(json.dumps(original_record, ensure_ascii=False) + "\n", encoding="utf-8")
    test_file.write_text("", encoding="utf-8")

    gop_du_lieu(val_file, test_file, out_file, dedup_key="qid", log_path=log_file)

    line = json.loads(out_file.read_text(encoding="utf-8").strip())

    for key, val in original_record.items():
        assert line[key] == val, f"Trường {key} bị thay đổi!"

    assert line["nguon_goc"] == "val"
    assert set(line.keys()) == set(list(original_record.keys()) + ["nguon_goc"])


def test_cung_qid_khac_noi_dung_duoc_giu_ca_hai(tmp_path: Path):
    """Cùng qid nhưng nội dung khác ở hai nguồn => cả hai dòng đều được giữ (hash nội dung)."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    val_data = [{"qid": 7, "asin": "A1", "questionText": "Cai gi do?"}]
    test_data = [{"qid": 7, "asin": "B2", "questionText": "Khac hoan toan?"}]

    val_file.write_text("\n".join(json.dumps(d) for d in val_data) + "\n", encoding="utf-8")
    test_file.write_text("\n".join(json.dumps(d) for d in test_data) + "\n", encoding="utf-8")

    stats = gop_du_lieu(val_file, test_file, out_file, dedup_key="hash_noi_dung", log_path=log_file)

    assert stats["dedup_dropped"] == 0
    assert stats["total_written"] == 2

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2
    uids = sorted(d["uid"] for d in lines)
    assert uids == ["test-7", "val-7"]
    val_line = [d for d in lines if d["nguon_goc"] == "val"][0]
    assert val_line["asin"] == "A1" and val_line["questionText"] == "Cai gi do?"


def test_hai_dong_giong_nhau_noi_dung_chi_giu_mot(tmp_path: Path):
    """Hai dòng giống hệt nội dung (bao gồm cả qid) => chỉ giữ một dòng đầu tiên (hash nội dung)."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    out_file = tmp_path / "out.jsonl"
    log_file = tmp_path / "test.log"

    val_data = [{"qid": 5, "asin": "A1", "questionText": "Noi dung giong nhau?"}]
    test_data = [{"qid": 5, "asin": "A1", "questionText": "Noi dung giong nhau?"}]

    val_file.write_text("\n".join(json.dumps(d) for d in val_data) + "\n", encoding="utf-8")
    test_file.write_text("\n".join(json.dumps(d) for d in test_data) + "\n", encoding="utf-8")

    stats = gop_du_lieu(val_file, test_file, out_file, dedup_key="hash_noi_dung", log_path=log_file)

    assert stats["dedup_dropped"] == 1
    assert stats["total_written"] == 1

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 1
    assert lines[0]["nguon_goc"] == "val"
    assert lines[0]["uid"] == "val-5"
