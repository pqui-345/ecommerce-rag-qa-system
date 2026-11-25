"""Kiểm thử cho module phân đoạn cơ sở src/chunking.py."""

import json
from pathlib import Path
import pytest

from src.chunking import chia_doan_fixed_size, tao_chunks_tu_file_docs


def test_chia_doan_fixed_size_co_ban():
    words = [f"w{i}" for i in range(10)]
    text = " ".join(words)
    chunks = chia_doan_fixed_size(text, chunk_size_words=4, chunk_overlap_words=1)
    assert len(chunks) == 3
    assert chunks[0]["text"] == "w0 w1 w2 w3"
    assert chunks[0]["start_word"] == 0
    assert chunks[0]["end_word"] == 4
    assert chunks[1]["text"] == "w3 w4 w5 w6"
    assert chunks[1]["start_word"] == 3
    assert chunks[1]["end_word"] == 7
    assert chunks[2]["text"] == "w6 w7 w8 w9"
    assert chunks[2]["start_word"] == 6
    assert chunks[2]["end_word"] == 10


def test_chia_doan_fixed_size_giu_nguyen_ky_hieu_va_xuong_dong():
    text = "Battery lasts <10 hours.\n\nDimensions: 5*3 inches > standard."
    chunks = chia_doan_fixed_size(text, chunk_size_words=20, chunk_overlap_words=5)
    assert len(chunks) == 1
    assert "\n\n" in chunks[0]["text"]
    assert "<10" in chunks[0]["text"]
    assert "5*3" in chunks[0]["text"]
    assert chunks[0]["n_words"] == 9


def test_chia_doan_fixed_size_rong_va_tham_so_loi():
    assert chia_doan_fixed_size("", chunk_size_words=10, chunk_overlap_words=2) == []
    assert chia_doan_fixed_size("   \n\n  ", chunk_size_words=10, chunk_overlap_words=2) == []
    with pytest.raises(ValueError):
        chia_doan_fixed_size("hello world", chunk_size_words=0, chunk_overlap_words=0)
    with pytest.raises(ValueError):
        chia_doan_fixed_size("hello world", chunk_size_words=5, chunk_overlap_words=5)
    with pytest.raises(ValueError):
        chia_doan_fixed_size("hello world", chunk_size_words=5, chunk_overlap_words=-1)


def test_tao_chunks_tu_file_docs_dong_hong_va_file_rong(tmp_path: Path):
    empty_in = tmp_path / "empty.jsonl"
    empty_out = tmp_path / "empty_out.jsonl"
    empty_in.write_text("", encoding="utf-8")

    kq_empty = tao_chunks_tu_file_docs(empty_in, empty_out, chunk_size_words=5, chunk_overlap_words=1)
    assert kq_empty["n_docs_in"] == 0
    assert kq_empty["n_chunks_out"] == 0
    assert empty_out.read_text(encoding="utf-8") == ""

    corrupt_in = tmp_path / "docs.jsonl"
    corrupt_out = tmp_path / "chunks.jsonl"
    lines = [
        json.dumps(
            {
                "doc_id": "doc-A1",
                "asin": "A1",
                "category": "Electronics",
                "nguon_goc": "val",
                "text_clean": "one two three four five six seven",
            }
        ),
        "{dòng json bị hỏng",
        json.dumps(
            {
                "doc_id": "doc-A2",
                "asin": "A2",
                "category": "Electronics",
                "nguon_goc": "val",
                "text_clean": "   ",
            }
        ),
    ]
    corrupt_in.write_text("\n".join(lines) + "\n", encoding="utf-8")

    kq = tao_chunks_tu_file_docs(corrupt_in, corrupt_out, chunk_size_words=4, chunk_overlap_words=1)
    assert kq["n_docs_in"] == 2
    assert kq["n_corrupt"] == 1
    assert kq["n_empty_docs"] == 1
    assert kq["n_chunks_out"] == 2
    assert kq["kiem_chunk_id_duy_nhat"] is True

    out_lines = [json.loads(line) for line in corrupt_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(out_lines) == 2
    assert out_lines[0]["chunk_id"] == "doc-A1#chunk_0000"
    assert out_lines[1]["chunk_id"] == "doc-A1#chunk_0001"


from src.chunking import (
    chay_chunk_strategies,
    chia_doan_parent_child,
    chia_doan_recursive,
    chia_doan_semantic,
    chia_doan_sentence,
    chia_doan_structure,
    tao_chunks_chien_luoc,
    thong_ke_ranh_gioi,
)


def _fake_embed(texts):
    vecs = []
    for t in texts:
        tl = t.lower()
        if "battery" in tl or "power" in tl:
            vecs.append([1.0, 0.0, 0.0])
        elif "screen" in tl or "display" in tl:
            vecs.append([0.0, 1.0, 0.0])
        else:
            vecs.append([0.0, 0.0, 1.0])
    return vecs


def test_chia_doan_recursive_sentence_structure():
    p1 = "Battery lasts <10 hours on 5*3 mode. Power adapter is 12V DC."
    p2 = "Screen resolution is 1080p HD! Display brightness > 300 nits?"
    p3 = "USB port supports 64GB flash drive. WiFi works well."
    text = f"{p1}\n\n{p2}\n\n{p3}"

    rec_chunks = chia_doan_recursive(text, max_words=15, overlap_words=4)
    assert len(rec_chunks) >= 2
    assert all(1 <= c["n_words"] <= 15 for c in rec_chunks)
    assert "<10" in rec_chunks[0]["text"] and "5*3" in rec_chunks[0]["text"]

    sent_chunks = chia_doan_sentence(text, max_words=15, overlap_words=5)
    assert len(sent_chunks) >= 2
    assert all(1 <= c["n_words"] <= 15 for c in sent_chunks)
    st_sent = thong_ke_ranh_gioi(text, [(c["start_char"], c["end_char"]) for c in sent_chunks])
    assert st_sent["n_start_ok"] == st_sent["n_chunks"]
    assert st_sent["n_end_ok"] == st_sent["n_chunks"]

    struct_chunks = chia_doan_structure(text, max_words=15)
    assert len(struct_chunks) == 3
    assert [c["text"] for c in struct_chunks] == [p1, p2, p3]
    st_struct = thong_ke_ranh_gioi(text, [(c["start_char"], c["end_char"]) for c in struct_chunks])
    assert st_struct["n_reviews_intact"] == 3


def test_chia_doan_semantic_va_parent_child():
    text = (
        "Battery life is great for travel. Power consumption is very low. "
        "Battery charges in two hours.\n\n"
        "Screen colors are vivid and bright. Display viewing angle is wide. "
        "Screen has anti glare coating."
    )
    sem_chunks = chia_doan_semantic(
        text,
        embed_fn=_fake_embed,
        max_words=30,
        min_words=10,
        percentile=70.0,
        buffer_size=0,
    )
    assert len(sem_chunks) == 2
    assert "Battery" in sem_chunks[0]["text"]
    assert "Screen" in sem_chunks[1]["text"]

    pc_chunks = chia_doan_parent_child(text, parent_max_words=25, child_max_words=10)
    assert len(pc_chunks) == 2
    assert all(len(p["children"]) >= 2 for p in pc_chunks)
    assert all(c["n_words"] <= 10 for p in pc_chunks for c in p["children"])


def test_tao_chunks_chien_luoc_va_chay_chunk_strategies(tmp_path: Path):
    docs_path = tmp_path / "docs_dev_clean.jsonl"
    text = (
        "First review sentence one. First review sentence two is here.\n\n"
        "Second review talks about <10 hours battery and 5*3 size."
    )
    lines = [
        json.dumps(
            {
                "doc_id": "doc-B1",
                "asin": "B1",
                "category": "Electronics",
                "nguon_goc": "val",
                "text_clean": text,
            }
        ),
        "{corrupt json",
        json.dumps({"doc_id": "doc-B2", "asin": "B2", "text_clean": "   "}),
    ]
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_path = tmp_path / "stats.md"
    res = chay_chunk_strategies(
        processed_dir=tmp_path,
        report_path=report_path,
        strategies=("recursive", "sentence", "structure", "semantic", "parent_child"),
        params={"max_words": 15, "overlap_words": 3, "child_max_words": 7, "semantic_min_words": 5},
        embed_fn=_fake_embed,
        split="dev",
    )
    assert report_path.exists()
    assert len(res["results"]) == 6
    child_file = tmp_path / "chunks_dev_parent_child_children.jsonl"
    assert child_file.exists()
    children = [json.loads(l) for l in child_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(children) >= 2
    assert all(c["parent_id"].startswith("doc-B1#pc_parent_") for c in children)
