"""Module phân đoạn văn bản: Fixed-size (baseline) và 5 chiến lược nâng cao."""

import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
WORD_RE = re.compile(r"\S+")
PARA_SEP_RE = re.compile(r"\n\s*\n")
SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
SENT_END_CHARS = set(".!?\"')]")

Span = Tuple[int, int]
EmbedFn = Callable[[Sequence[str]], List[List[float]]]


def _percentile_sorted(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (q / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _thong_ke_day_so(vals: List[int]) -> Dict[str, float]:
    if not vals:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0}
    s = sorted(vals)
    return {
        "min": int(s[0]),
        "median": round(float(statistics.median(s)), 1),
        "mean": round(float(statistics.mean(s)), 1),
        "p90": round(float(_percentile_sorted(s, 90)), 1),
        "max": int(s[-1]),
    }


def chia_doan_fixed_size(
    text: str,
    chunk_size_words: int = 200,
    chunk_overlap_words: int = 40,
) -> List[Dict]:
    """Chia văn bản theo cửa sổ trượt cố định số từ (không cắt đôi từ, giữ nguyên \\n\\n)."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words phải lớn hơn 0.")
    if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words phải nằm trong khoảng [0, chunk_size_words).")

    if not isinstance(text, str) or not text.strip():
        return []

    matches = list(WORD_RE.finditer(text))
    total_words = len(matches)
    if total_words == 0:
        return []

    step = chunk_size_words - chunk_overlap_words
    chunks: List[Dict] = []
    start_w = 0
    chunk_idx = 0

    while start_w < total_words:
        end_w = min(start_w + chunk_size_words, total_words)
        start_char = matches[start_w].start()
        end_char = matches[end_w - 1].end()
        chunk_text = text[start_char:end_char]
        n_words = end_w - start_w

        chunks.append(
            {
                "chunk_index": chunk_idx,
                "start_word": start_w,
                "end_word": end_w,
                "start_char": start_char,
                "end_char": end_char,
                "n_words": n_words,
                "n_chars": len(chunk_text),
                "text": chunk_text,
            }
        )
        chunk_idx += 1

        if end_w >= total_words:
            break
        start_w += step

    return chunks


def tao_chunks_tu_file_docs(
    input_path: Path,
    output_path: Path,
    chunk_size_words: int = 200,
    chunk_overlap_words: int = 40,
    strategy_name: str = "fixed_size",
) -> Dict:
    """Đọc streaming docs_*_clean.jsonl và xuất chunks_*_fixed.jsonl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_docs_in = 0
    n_corrupt = 0
    n_empty_docs = 0
    n_chunks_out = 0
    chunk_ids_seen = set()
    dup_chunk_ids = 0

    chunks_per_doc: List[int] = []
    words_per_chunk: List[int] = []
    chars_per_chunk: List[int] = []

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                n_corrupt += 1
                continue

            if not isinstance(doc, dict):
                n_corrupt += 1
                continue

            n_docs_in += 1
            doc_id = str(doc.get("doc_id") or f"doc-{n_docs_in}")
            asin = str(doc.get("asin") or "")
            category = str(doc.get("category") or "")
            nguon_goc = str(doc.get("nguon_goc") or "")
            text = doc.get("text_clean")
            if text is None:
                text = doc.get("text_goc", "")

            doc_chunks = chia_doan_fixed_size(
                text=text,
                chunk_size_words=chunk_size_words,
                chunk_overlap_words=chunk_overlap_words,
            )
            if not doc_chunks:
                n_empty_docs += 1
                chunks_per_doc.append(0)
                continue

            total_in_doc = len(doc_chunks)
            chunks_per_doc.append(total_in_doc)

            for ch in doc_chunks:
                idx = ch["chunk_index"]
                chunk_id = f"{doc_id}#chunk_{idx:04d}"
                if chunk_id in chunk_ids_seen:
                    dup_chunk_ids += 1
                chunk_ids_seen.add(chunk_id)

                rec = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "asin": asin,
                    "category": category,
                    "nguon_goc": nguon_goc,
                    "strategy": strategy_name,
                    "chunk_index": idx,
                    "total_chunks_in_doc": total_in_doc,
                    "start_word": ch["start_word"],
                    "end_word": ch["end_word"],
                    "n_words": ch["n_words"],
                    "n_chars": ch["n_chars"],
                    "text": ch["text"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks_out += 1
                words_per_chunk.append(ch["n_words"])
                chars_per_chunk.append(ch["n_chars"])

    return {
        "n_docs_in": n_docs_in,
        "n_corrupt": n_corrupt,
        "n_empty_docs": n_empty_docs,
        "n_chunks_out": n_chunks_out,
        "kiem_chunk_id_duy_nhat": dup_chunk_ids == 0 and n_chunks_out == len(chunk_ids_seen),
        "stats_chunks_per_doc": _thong_ke_day_so(chunks_per_doc),
        "stats_words_per_chunk": _thong_ke_day_so(words_per_chunk),
        "stats_chars_per_chunk": _thong_ke_day_so(chars_per_chunk),
    }


def chay_chunk_baseline(
    processed_dir: Path,
    report_path: Path,
    chunk_size_words: int = 200,
    chunk_overlap_words: int = 40,
) -> Dict:
    """Thực thi phân đoạn Fixed-size cho cả tập dev và test, xuất báo cáo Markdown."""
    dev_in = processed_dir / "docs_dev_clean.jsonl"
    test_in = processed_dir / "docs_test_clean.jsonl"
    dev_out = processed_dir / "chunks_dev_fixed.jsonl"
    test_out = processed_dir / "chunks_test_fixed.jsonl"

    kq_dev = tao_chunks_tu_file_docs(
        input_path=dev_in,
        output_path=dev_out,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
        strategy_name="fixed_size",
    )
    kq_test = tao_chunks_tu_file_docs(
        input_path=test_in,
        output_path=test_out,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
        strategy_name="fixed_size",
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Phân đoạn Cơ sở (Baseline Fixed-size Chunking)",
        "",
        "## 1. Cấu hình tham số",
        "- Chiến lược: `fixed_size` (cửa sổ trượt theo ranh giới từ, không cắt đôi từ, bảo toàn `\\n\\n` nội bộ)",
        f"- Kích thước đoạn (`chunk_size_words`): **{chunk_size_words} từ** (~250-280 tokens, tương thích hoàn toàn với giới hạn 512 tokens của `BAAI/bge-small-en-v1.5`)",
        f"- Độ gối đầu (`chunk_overlap_words`): **{chunk_overlap_words} từ** ({round(chunk_overlap_words * 100.0 / chunk_size_words, 1)}%)",
        f"- Bước trượt (`step`): **{chunk_size_words - chunk_overlap_words} từ**",
        "",
        "## 2. Thống kê kết quả phân đoạn",
        "",
        "| Chỉ số | Tập Dev (`chunks_dev_fixed.jsonl`) | Tập Test (`chunks_test_fixed.jsonl`) |",
        "| :--- | ---: | ---: |",
        f"| Số tài liệu đầu vào (`docs_*_clean.jsonl`) | {kq_dev['n_docs_in']} | {kq_test['n_docs_in']} |",
        f"| Số dòng lỗi JSON / tài liệu rỗng | {kq_dev['n_corrupt']} / {kq_dev['n_empty_docs']} | {kq_test['n_corrupt']} / {kq_test['n_empty_docs']} |",
        f"| **Tổng số đoạn văn (chunks) sinh ra** | **{kq_dev['n_chunks_out']}** | **{kq_test['n_chunks_out']}** |",
        f"| Kiểm tra `chunk_id` duy nhất 100% | {'ĐẠT' if kq_dev['kiem_chunk_id_duy_nhat'] else 'KHÔNG ĐẠT'} | {'ĐẠT' if kq_test['kiem_chunk_id_duy_nhat'] else 'KHÔNG ĐẠT'} |",
        f"| Số chunks / tài liệu (Min / Trung vị / TB / P90 / Max) | {kq_dev['stats_chunks_per_doc']['min']} / {kq_dev['stats_chunks_per_doc']['median']} / {kq_dev['stats_chunks_per_doc']['mean']} / {kq_dev['stats_chunks_per_doc']['p90']} / {kq_dev['stats_chunks_per_doc']['max']} | {kq_test['stats_chunks_per_doc']['min']} / {kq_test['stats_chunks_per_doc']['median']} / {kq_test['stats_chunks_per_doc']['mean']} / {kq_test['stats_chunks_per_doc']['p90']} / {kq_test['stats_chunks_per_doc']['max']} |",
        f"| Số từ / chunk (Min / Trung vị / TB / P90 / Max) | {kq_dev['stats_words_per_chunk']['min']} / {kq_dev['stats_words_per_chunk']['median']} / {kq_dev['stats_words_per_chunk']['mean']} / {kq_dev['stats_words_per_chunk']['p90']} / {kq_dev['stats_words_per_chunk']['max']} | {kq_test['stats_words_per_chunk']['min']} / {kq_test['stats_words_per_chunk']['median']} / {kq_test['stats_words_per_chunk']['mean']} / {kq_test['stats_words_per_chunk']['p90']} / {kq_test['stats_words_per_chunk']['max']} |",
        f"| Số ký tự / chunk (Min / Trung vị / TB / P90 / Max) | {kq_dev['stats_chars_per_chunk']['min']} / {kq_dev['stats_chars_per_chunk']['median']} / {kq_dev['stats_chars_per_chunk']['mean']} / {kq_dev['stats_chars_per_chunk']['p90']} / {kq_dev['stats_chars_per_chunk']['max']} | {kq_test['stats_chars_per_chunk']['min']} / {kq_test['stats_chars_per_chunk']['median']} / {kq_test['stats_chars_per_chunk']['mean']} / {kq_test['stats_chars_per_chunk']['p90']} / {kq_test['stats_chars_per_chunk']['max']} |",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {"dev": kq_dev, "test": kq_test, "report_path": str(report_path)}



def _dem_tu(text: str, start: int, end: int) -> int:
    return len(WORD_RE.findall(text, start, end))


def _cat_khoang_trang(text: str, start: int, end: int) -> Optional[Span]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def tach_doan_nhan_xet(text: str, start: int = 0, end: Optional[int] = None) -> List[Span]:
    """Tách khoảng [start, end) thành các đoạn nhận xét theo ranh giới dòng trống."""
    end = len(text) if end is None else end
    spans: List[Span] = []
    pos = start
    for m in PARA_SEP_RE.finditer(text, start, end):
        sp = _cat_khoang_trang(text, pos, m.start())
        if sp:
            spans.append(sp)
        pos = m.end()
    sp = _cat_khoang_trang(text, pos, end)
    if sp:
        spans.append(sp)
    return spans


def tach_cau(text: str, start: int, end: int) -> List[Span]:
    """Tách câu trong một đoạn nhận xét theo dấu kết câu .!? theo sau là khoảng trắng."""
    spans: List[Span] = []
    pos = start
    for m in SENT_END_RE.finditer(text, start, end):
        sp = _cat_khoang_trang(text, pos, m.end())
        if sp:
            spans.append(sp)
        pos = m.end()
    sp = _cat_khoang_trang(text, pos, end)
    if sp:
        spans.append(sp)
    return spans


def tach_cau_tai_lieu(text: str, start: int = 0, end: Optional[int] = None) -> List[Span]:
    """Tách câu trên toàn khoảng, câu không bao giờ vượt qua ranh giới nhận xét."""
    out: List[Span] = []
    for ps, pe in tach_doan_nhan_xet(text, start, end):
        out.extend(tach_cau(text, ps, pe))
    return out


def _cat_theo_cua_so_tu(text: str, span: Span, max_words: int) -> List[Span]:
    words = list(WORD_RE.finditer(text, span[0], span[1]))
    out: List[Span] = []
    for i in range(0, len(words), max_words):
        grp = words[i : i + max_words]
        out.append((grp[0].start(), grp[-1].end()))
    return out


def _gioi_han_don_vi(text: str, spans: Sequence[Span], max_words: int) -> List[Span]:
    out: List[Span] = []
    for sp in spans:
        if _dem_tu(text, sp[0], sp[1]) > max_words:
            out.extend(_cat_theo_cua_so_tu(text, sp, max_words))
        else:
            out.append(sp)
    return out


def gom_don_vi(
    text: str,
    units: Sequence[Span],
    max_words: int,
    overlap_words: int = 0,
) -> List[Span]:
    """Gom tham lam các đơn vị liên tiếp thành đoạn <= max_words, gối đầu theo đơn vị.

    Đơn vị gối đầu được lấy từ cuối đoạn trước sao cho tổng số từ gối đầu
    <= overlap_words và đoạn kế tiếp vẫn chứa ít nhất một đơn vị mới.
    """
    if max_words <= 0:
        raise ValueError("max_words phải lớn hơn 0.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words phải nằm trong khoảng [0, max_words).")
    n = len(units)
    if n == 0:
        return []
    counts = [_dem_tu(text, s, e) for s, e in units]
    chunks: List[Span] = []
    i = 0
    while i < n:
        cur = 0
        j = i
        while j < n and (j == i or cur + counts[j] <= max_words):
            cur += counts[j]
            j += 1
        chunks.append((units[i][0], units[j - 1][1]))
        if j >= n:
            break
        k = j
        ov = 0
        while k - 1 > i and ov + counts[k - 1] <= overlap_words and ov + counts[k - 1] + counts[j] <= max_words:
            ov += counts[k - 1]
            k -= 1
        i = k
    return chunks


def _tach_de_quy(
    text: str,
    start: int,
    end: int,
    separators: Sequence[str],
    max_words: int,
) -> List[Span]:
    sp = _cat_khoang_trang(text, start, end)
    if not sp:
        return []
    start, end = sp
    if _dem_tu(text, start, end) <= max_words:
        return [(start, end)]
    for idx, sep in enumerate(separators):
        occ = text.find(sep, start, end)
        if occ == -1:
            continue
        pieces: List[Span] = []
        pos = start
        while occ != -1:
            cut = occ + len(sep)
            pieces.append((pos, cut))
            pos = cut
            occ = text.find(sep, cut, end)
        pieces.append((pos, end))
        out: List[Span] = []
        for ps, pe in pieces:
            out.extend(_tach_de_quy(text, ps, pe, separators[idx + 1 :], max_words))
        return out
    return _cat_theo_cua_so_tu(text, (start, end), max_words)


def _tao_ban_ghi(text: str, spans: Sequence[Span]) -> List[Dict]:
    out: List[Dict] = []
    for idx, (s, e) in enumerate(spans):
        chunk_text = text[s:e]
        out.append(
            {
                "chunk_index": idx,
                "start_char": s,
                "end_char": e,
                "n_words": _dem_tu(text, s, e),
                "n_chars": len(chunk_text),
                "text": chunk_text,
            }
        )
    return out


def _van_ban_hop_le(text) -> bool:
    return isinstance(text, str) and bool(text.strip())


def chia_doan_recursive(
    text: str,
    max_words: int = 200,
    overlap_words: int = 40,
    separators: Sequence[str] = RECURSIVE_SEPARATORS,
) -> List[Dict]:
    """Tách đệ quy theo danh sách dấu phân cách ưu tiên rồi gom lại <= max_words."""
    if not _van_ban_hop_le(text):
        return []
    pieces = _tach_de_quy(text, 0, len(text), list(separators), max_words)
    return _tao_ban_ghi(text, gom_don_vi(text, pieces, max_words, overlap_words))


def chia_doan_sentence(text: str, max_words: int = 200, overlap_words: int = 40) -> List[Dict]:
    """Gom các câu hoàn chỉnh liên tiếp <= max_words, gối đầu theo câu."""
    if not _van_ban_hop_le(text):
        return []
    units = _gioi_han_don_vi(text, tach_cau_tai_lieu(text), max_words)
    return _tao_ban_ghi(text, gom_don_vi(text, units, max_words, overlap_words))


def _don_vi_structure(text: str, start: int, end: int, max_words: int) -> List[Span]:
    units: List[Span] = []
    for ps, pe in tach_doan_nhan_xet(text, start, end):
        if _dem_tu(text, ps, pe) <= max_words:
            units.append((ps, pe))
        else:
            sents = _gioi_han_don_vi(text, tach_cau(text, ps, pe), max_words)
            units.extend(gom_don_vi(text, sents, max_words, 0))
    return units


def chia_doan_structure(text: str, max_words: int = 200) -> List[Dict]:
    """Gom nguyên vẹn các đoạn nhận xet (\\n\\n) <= max_words; nhận xét quá dài tách theo câu."""
    if not _van_ban_hop_le(text):
        return []
    units = _don_vi_structure(text, 0, len(text), max_words)
    return _tao_ban_ghi(text, gom_don_vi(text, units, max_words, 0))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def chia_doan_semantic(
    text: str,
    embed_fn: EmbedFn,
    max_words: int = 200,
    min_words: int = 50,
    percentile: float = 90.0,
    buffer_size: int = 0,
) -> List[Dict]:
    """Ngắt đoạn tại điểm khoảng cách cosine giữa hai cụm câu liền kề vượt phân vị ngưỡng.

    Các câu quá ngắn trong cùng nhận xét được gom thành cụm nguyên tử (<= min(45, max_words//2) từ)
    để tránh nhiễu vector ở câu 1-3 từ; chỉ ngắt khi đoạn hiện tại đã >= min_words.
    """
    if not _van_ban_hop_le(text):
        return []
    unit_cap = min(45, max(10, max_words // 3))
    units: List[Span] = []
    for ps, pe in tach_doan_nhan_xet(text):
        sents = _gioi_han_don_vi(text, tach_cau(text, ps, pe), max_words)
        units.extend(gom_don_vi(text, sents, unit_cap, 0))
    n = len(units)
    if n < 3:
        return _tao_ban_ghi(text, gom_don_vi(text, units, max_words, 0))

    windows = [
        text[units[max(0, i - buffer_size)][0] : units[min(n - 1, i + buffer_size)][1]]
        for i in range(n)
    ]
    vecs = embed_fn(windows)
    dists = [1.0 - _cosine(vecs[i], vecs[i + 1]) for i in range(n - 1)]
    threshold = _percentile_sorted(sorted(dists), percentile)
    counts = [_dem_tu(text, s, e) for s, e in units]

    spans: List[Span] = []
    cs = 0
    cw = counts[0]
    for i in range(1, n):
        if cw + counts[i] > max_words or (dists[i - 1] > threshold and cw >= min_words):
            spans.append((units[cs][0], units[i - 1][1]))
            cs = i
            cw = counts[i]
        else:
            cw += counts[i]
    spans.append((units[cs][0], units[-1][1]))
    return _tao_ban_ghi(text, spans)


def chia_doan_parent_child(
    text: str,
    parent_max_words: int = 200,
    child_max_words: int = 60,
) -> List[Dict]:
    """Đoạn cha = structure-aware; đoạn con = gom câu <= child_max_words bên trong mỗi cha."""
    if not _van_ban_hop_le(text):
        return []
    parents = chia_doan_structure(text, parent_max_words)
    for p in parents:
        sents = _gioi_han_don_vi(
            text, tach_cau_tai_lieu(text, p["start_char"], p["end_char"]), child_max_words
        )
        child_spans = gom_don_vi(text, sents, child_max_words, 0)
        p["children"] = _tao_ban_ghi(text, child_spans)
    return parents


def thong_ke_ranh_gioi(text: str, spans: Sequence[Span]) -> Dict[str, int]:
    """Đếm số đoạn bắt đầu/kết thúc đúng ranh giới câu và số nhận xét bị cắt."""
    paras = tach_doan_nhan_xet(text)
    sents = tach_cau_tai_lieu(text)
    starts_ok = {s for s, _ in sents} | {s for s, _ in paras}
    ends_para = {e for _, e in paras}
    n_start_ok = 0
    n_end_ok = 0
    for s, e in spans:
        if s in starts_ok:
            n_start_ok += 1
        if e in ends_para or (e > 0 and text[e - 1] in SENT_END_CHARS):
            n_end_ok += 1
    sorted_spans = sorted(spans)
    n_intact = 0
    for ps, pe in paras:
        if any(s <= ps and e >= pe for s, e in sorted_spans):
            n_intact += 1
    return {
        "n_chunks": len(spans),
        "n_start_ok": n_start_ok,
        "n_end_ok": n_end_ok,
        "n_reviews": len(paras),
        "n_reviews_intact": n_intact,
    }


STRATEGIES_NANG_CAO = ("recursive", "sentence", "structure", "semantic", "parent_child")


def chia_doan_theo_chien_luoc(
    text: str,
    strategy: str,
    params: Optional[Dict] = None,
    embed_fn: Optional[EmbedFn] = None,
) -> List[Dict]:
    """Điều phối chiến lược phân đoạn theo tên."""
    p = params or {}
    max_words = int(p.get("max_words", 200))
    overlap = int(p.get("overlap_words", 40))
    if strategy == "fixed":
        return chia_doan_fixed_size(text, max_words, overlap)
    if strategy == "recursive":
        return chia_doan_recursive(text, max_words, overlap)
    if strategy == "sentence":
        return chia_doan_sentence(text, max_words, overlap)
    if strategy == "structure":
        return chia_doan_structure(text, max_words)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("Chiến lược semantic cần embed_fn.")
        return chia_doan_semantic(
            text,
            embed_fn,
            max_words=max_words,
            min_words=int(p.get("semantic_min_words", 50)),
            percentile=float(p.get("semantic_percentile", 90.0)),
            buffer_size=int(p.get("semantic_buffer_size", 0)),
        )
    if strategy == "parent_child":
        return chia_doan_parent_child(text, max_words, int(p.get("child_max_words", 60)))
    raise ValueError(f"Chiến lược không hỗ trợ: {strategy}")


def tao_chunks_chien_luoc(
    input_path: Path,
    output_path: Path,
    strategy: str,
    params: Optional[Dict] = None,
    embed_fn: Optional[EmbedFn] = None,
    children_output_path: Optional[Path] = None,
) -> Dict:
    """Đọc streaming docs_*_clean.jsonl, phân đoạn theo chiến lược và ghi chunks JSONL.

    Với parent_child: output_path chứa đoạn cha (đơn vị đưa vào LLM),
    children_output_path chứa đoạn con (đơn vị truy xuất) kèm parent_id.
    """
    if strategy == "parent_child" and children_output_path is None:
        raise ValueError("parent_child cần children_output_path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_docs_in = 0
    n_corrupt = 0
    n_empty_docs = 0
    n_chunks_out = 0
    n_children_out = 0
    ids_seen = set()
    n_dup = 0
    chunks_per_doc: List[int] = []
    words_per_chunk: List[int] = []
    words_per_child: List[int] = []
    bd = {"n_chunks": 0, "n_start_ok": 0, "n_end_ok": 0, "n_reviews": 0, "n_reviews_intact": 0}

    fchild = open(children_output_path, "w", encoding="utf-8") if children_output_path else None
    try:
        with (
            open(input_path, "r", encoding="utf-8") as fin,
            open(output_path, "w", encoding="utf-8") as fout,
        ):
            for raw_line in fin:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    n_corrupt += 1
                    continue
                if not isinstance(doc, dict):
                    n_corrupt += 1
                    continue

                n_docs_in += 1
                doc_id = str(doc.get("doc_id") or f"doc-{n_docs_in}")
                text = doc.get("text_clean")
                if text is None:
                    text = doc.get("text_goc", "")
                base = {
                    "doc_id": doc_id,
                    "asin": str(doc.get("asin") or ""),
                    "category": str(doc.get("category") or ""),
                    "nguon_goc": str(doc.get("nguon_goc") or ""),
                    "strategy": strategy,
                }

                doc_chunks = chia_doan_theo_chien_luoc(text, strategy, params, embed_fn)
                if not doc_chunks:
                    n_empty_docs += 1
                    chunks_per_doc.append(0)
                    continue
                chunks_per_doc.append(len(doc_chunks))

                st = thong_ke_ranh_gioi(text, [(c["start_char"], c["end_char"]) for c in doc_chunks])
                for k in bd:
                    bd[k] += st[k]

                prefix = "pc_parent" if strategy == "parent_child" else strategy
                for ch in doc_chunks:
                    idx = ch["chunk_index"]
                    chunk_id = f"{doc_id}#{prefix}_{idx:04d}"
                    if chunk_id in ids_seen:
                        n_dup += 1
                    ids_seen.add(chunk_id)
                    rec = dict(base)
                    rec.update(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": idx,
                            "total_chunks_in_doc": len(doc_chunks),
                            "start_char": ch["start_char"],
                            "end_char": ch["end_char"],
                            "n_words": ch["n_words"],
                            "n_chars": ch["n_chars"],
                            "text": ch["text"],
                        }
                    )
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_chunks_out += 1
                    words_per_chunk.append(ch["n_words"])

                    for child in ch.get("children", []):
                        cidx = child["chunk_index"]
                        child_id = f"{doc_id}#pc_child_{idx:04d}_{cidx:02d}"
                        if child_id in ids_seen:
                            n_dup += 1
                        ids_seen.add(child_id)
                        crec = dict(base)
                        crec.update(
                            {
                                "chunk_id": child_id,
                                "parent_id": chunk_id,
                                "chunk_index": cidx,
                                "start_char": child["start_char"],
                                "end_char": child["end_char"],
                                "n_words": child["n_words"],
                                "n_chars": child["n_chars"],
                                "text": child["text"],
                            }
                        )
                        fchild.write(json.dumps(crec, ensure_ascii=False) + "\n")
                        n_children_out += 1
                        words_per_child.append(child["n_words"])
    finally:
        if fchild:
            fchild.close()

    return {
        "strategy": strategy,
        "n_docs_in": n_docs_in,
        "n_corrupt": n_corrupt,
        "n_empty_docs": n_empty_docs,
        "n_chunks_out": n_chunks_out,
        "n_children_out": n_children_out,
        "kiem_chunk_id_duy_nhat": n_dup == 0,
        "stats_chunks_per_doc": _thong_ke_day_so(chunks_per_doc),
        "stats_words_per_chunk": _thong_ke_day_so(words_per_chunk),
        "stats_words_per_child": _thong_ke_day_so(words_per_child),
        "boundary": bd,
    }


def _ty_le(a: int, b: int) -> float:
    return round(a * 100.0 / b, 2) if b else 0.0


def thong_ke_fixed_tu_docs(input_path: Path, max_words: int = 200, overlap_words: int = 40) -> Dict:
    """Tính thống kê ranh giới cho Fixed-size trên cùng tài liệu (không ghi file)."""
    bd = {"n_chunks": 0, "n_start_ok": 0, "n_end_ok": 0, "n_reviews": 0, "n_reviews_intact": 0}
    chunks_per_doc: List[int] = []
    words: List[int] = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text_clean") or ""
            chs = chia_doan_fixed_size(text, max_words, overlap_words)
            chunks_per_doc.append(len(chs))
            words.extend(c["n_words"] for c in chs)
            st = thong_ke_ranh_gioi(text, [(c["start_char"], c["end_char"]) for c in chs])
            for k in bd:
                bd[k] += st[k]
    return {
        "strategy": "fixed",
        "n_docs_in": len(chunks_per_doc),
        "n_chunks_out": sum(chunks_per_doc),
        "n_children_out": 0,
        "stats_chunks_per_doc": _thong_ke_day_so(chunks_per_doc),
        "stats_words_per_chunk": _thong_ke_day_so(words),
        "stats_words_per_child": _thong_ke_day_so([]),
        "boundary": bd,
        "kiem_chunk_id_duy_nhat": True,
        "n_corrupt": 0,
        "n_empty_docs": 0,
    }


def ghi_bao_cao_thong_ke_chien_luoc(results: Sequence[Dict], report_path: Path, params: Dict) -> None:
    """Ghi báo cáo Markdown thống kê và chất lượng ranh giới của các chiến lược."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Thống kê các chiến lược phân đoạn trên tập dev (150 tài liệu Electronics)",
        "",
        "## 1. Tham số",
        f"- Trần kích thước: **{params.get('max_words', 200)} từ/đoạn**; overlap (fixed, recursive, sentence): **{params.get('overlap_words', 40)} từ**",
        f"- Semantic: phân vị ngưỡng **P{params.get('semantic_percentile', 90.0):g}**, tối thiểu **{params.get('semantic_min_words', 50)} từ** trước khi ngắt, cửa sổ nhúng ±{params.get('semantic_buffer_size', 1)} câu",
        f"- Parent-child: cha = structure-aware ≤ {params.get('max_words', 200)} từ, con ≤ **{params.get('child_max_words', 60)} từ**",
        "",
        "## 2. Kích thước đoạn",
        "",
        "| Chiến lược | Số đoạn | Đoạn/tài liệu (min / trung vị / TB / max) | Từ/đoạn (min / trung vị / TB / P90 / max) | Đoạn con (số / từ TB) | chunk_id duy nhất |",
        "| :--- | ---: | :--- | :--- | :--- | :---: |",
    ]
    for r in results:
        cpd = r["stats_chunks_per_doc"]
        wpc = r["stats_words_per_chunk"]
        child = (
            f"{r['n_children_out']} / {r['stats_words_per_child']['mean']}"
            if r.get("n_children_out")
            else "-"
        )
        lines.append(
            f"| `{r['strategy']}` | {r['n_chunks_out']} | {cpd['min']} / {cpd['median']} / {cpd['mean']} / {cpd['max']} | "
            f"{wpc['min']} / {wpc['median']} / {wpc['mean']} / {wpc['p90']} / {wpc['max']} | {child} | "
            f"{'ĐẠT' if r['kiem_chunk_id_duy_nhat'] else 'KHÔNG ĐẠT'} |"
        )
    lines.extend(
        [
            "",
            "## 3. Chất lượng ranh giới đoạn",
            "",
            "| Chiến lược | % đoạn bắt đầu ở đầu câu/nhận xét | % đoạn kết thúc ở cuối câu/nhận xét | % nhận xét nằm trọn trong ít nhất 1 đoạn |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for r in results:
        b = r["boundary"]
        lines.append(
            f"| `{r['strategy']}` | {_ty_le(b['n_start_ok'], b['n_chunks'])}% | {_ty_le(b['n_end_ok'], b['n_chunks'])}% | "
            f"{_ty_le(b['n_reviews_intact'], b['n_reviews'])}% ({b['n_reviews_intact']}/{b['n_reviews']}) |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def chay_chunk_strategies(
    processed_dir: Path,
    report_path: Path,
    strategies: Sequence[str] = STRATEGIES_NANG_CAO,
    params: Optional[Dict] = None,
    embed_fn: Optional[EmbedFn] = None,
    split: str = "dev",
) -> Dict:
    """Sinh chunks cho các chiến lược nâng cao trên một tập và ghi báo cáo thống kê."""
    params = params or {}
    docs_in = processed_dir / f"docs_{split}_clean.jsonl"
    results: List[Dict] = [
        thong_ke_fixed_tu_docs(docs_in, int(params.get("max_words", 200)), int(params.get("overlap_words", 40)))
    ]
    for st in strategies:
        out_path = processed_dir / f"chunks_{split}_{st}.jsonl"
        child_path = processed_dir / f"chunks_{split}_parent_child_children.jsonl" if st == "parent_child" else None
        results.append(
            tao_chunks_chien_luoc(
                input_path=docs_in,
                output_path=out_path,
                strategy=st,
                params=params,
                embed_fn=embed_fn if st == "semantic" else None,
                children_output_path=child_path,
            )
        )

    existing = {r["strategy"] for r in results}
    for st in STRATEGIES_NANG_CAO:
        path = processed_dir / f"chunks_{split}_{st}.jsonl"
        if st not in existing and path.exists():
            results.append(_thong_ke_tu_file_chunks(path, st, docs_in, processed_dir, split))

    order = ["fixed", *STRATEGIES_NANG_CAO]
    results.sort(key=lambda r: order.index(r["strategy"]) if r["strategy"] in order else 99)
    ghi_bao_cao_thong_ke_chien_luoc(results, report_path, params)
    return {"results": results, "report_path": str(report_path)}


def _thong_ke_tu_file_chunks(path: Path, strategy: str, docs_in: Path, processed_dir: Path, split: str) -> Dict:
    texts: Dict[str, str] = {}
    with open(docs_in, "r", encoding="utf-8") as fin:
        for raw in fin:
            if raw.strip():
                d = json.loads(raw)
                texts[d["doc_id"]] = d.get("text_clean") or ""
    by_doc: Dict[str, List[Dict]] = {}
    ids = set()
    n_dup = 0
    with open(path, "r", encoding="utf-8") as fin:
        for raw in fin:
            if raw.strip():
                c = json.loads(raw)
                by_doc.setdefault(c["doc_id"], []).append(c)
                if c["chunk_id"] in ids:
                    n_dup += 1
                ids.add(c["chunk_id"])
    bd = {"n_chunks": 0, "n_start_ok": 0, "n_end_ok": 0, "n_reviews": 0, "n_reviews_intact": 0}
    words: List[int] = []
    for did, chs in by_doc.items():
        st = thong_ke_ranh_gioi(texts.get(did, ""), [(c["start_char"], c["end_char"]) for c in chs])
        for k in bd:
            bd[k] += st[k]
        words.extend(c["n_words"] for c in chs)
    child_words: List[int] = []
    child_path = processed_dir / f"chunks_{split}_parent_child_children.jsonl"
    if strategy == "parent_child" and child_path.exists():
        with open(child_path, "r", encoding="utf-8") as fin:
            for raw in fin:
                if raw.strip():
                    child_words.append(json.loads(raw)["n_words"])
    return {
        "strategy": strategy,
        "n_docs_in": len(by_doc),
        "n_chunks_out": sum(len(v) for v in by_doc.values()),
        "n_children_out": len(child_words),
        "stats_chunks_per_doc": _thong_ke_day_so([len(v) for v in by_doc.values()]),
        "stats_words_per_chunk": _thong_ke_day_so(words),
        "stats_words_per_child": _thong_ke_day_so(child_words),
        "boundary": bd,
        "kiem_chunk_id_duy_nhat": n_dup == 0,
        "n_corrupt": 0,
        "n_empty_docs": 0,
    }
