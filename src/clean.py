"""Làm sạch văn bản cho docs_dev.jsonl / docs_test.jsonl.

Quy ước theo AGENTS.md:
- Không ghi đè file gốc; file mới ghi vào data/processed/.
- Đường dẫn tính bằng pathlib từ vị trí file.
- Hàm clean_text idempotent (clean_text(clean_text(x)) == clean_text(x)).

Quy tắc làm sạch (đã chốt theo nhiệm vụ):
1. Giải mã thực thể HTML (html.unescape) dùng tới khi hội tụ (lặp tối đa 5 lần).
2. NFKC; xóa zero-width; NBSP và khoảng trắng unicode lạ -> khoảng trắng ' ';
   smart quotes -> ' và ".
3. \\r\\n và \\r -> \\n; \\n đơn (không thuộc cặp \\n\\n) -> khoảng trắng;
   dãy \\n>=3 -> chính xác \\n\\n (giữ ranh giới đoạn theo AGENTS.md).
4. Chỉ xóa thẻ HTML có tên nằm trong danh sách thẻ cho phép (không dùng <.*?>).
5. Giữ nguyên * < > trong nội dung thường.
6. Gộp 2+ khoảng trắng thành 1, strip hai đầu. Không đổi hoa/thường,
   không xóa dấu câu, không xóa số.
"""

import html
import json
import random
import re
import statistics
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

# Danh sách thẻ HTML cho phép xóa (tên thẻ, chữ thường). KHÔNG dùng regex <.*?>.
ALLOWED_TAGS = [
    "a", "b", "big", "blockquote", "br", "center", "cite", "code", "div", "em",
    "font", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "li", "ol",
    "p", "pre", "s", "small", "span", "strike", "strong", "sub", "sup",
    "table", "tbody", "td", "tfoot", "th", "thead", "tr", "tt", "u", "ul",
]
_TAG_PATTERN = r"</?(?:" + "|".join(sorted(ALLOWED_TAGS)) + r")\b(?:\s[^<>]*)?/?>"
ALLOWED_TAG_RE = re.compile(_TAG_PATTERN, re.IGNORECASE)

RESIDUE_NUM_RE = re.compile(r"&#\d+;|&#x[0-9A-Fa-f]+;")
RESIDUE_NAMED_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|copy|reg);")

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
UNICODE_SPACES = ["\u00a0", "\u2007", "\u202f", "\u205f", "\u3000"]
LINE_SEPARATORS = ["\u2028", "\u2029", "\u0085", "\u000b", "\u000c"]

DONT_KNOW_RE = re.compile(
    r"(don['’]t|dont|do not)\s+know|idk|not\s+sure|no\s+idea",
    re.IGNORECASE,
)


def is_dont_know_answer(answer_text) -> bool:
    """Kiểm tra câu trả lời có phải dạng 'I don't know / not sure / no idea'."""
    if not isinstance(answer_text, str):
        return False
    return bool(DONT_KNOW_RE.search(answer_text))


def clean_text(raw) -> str:
    """Làm sạch một chuỗi văn bản theo quy tắc 1-6 ở trên. Idempotent."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    t = raw

    for _ in range(5):
        t2 = html.unescape(t)
        if t2 == t:
            break
        t = t2

    t = unicodedata.normalize("NFKC", t)

    for ch in ZERO_WIDTH:
        t = t.replace(ch, "")
    for ch in UNICODE_SPACES:
        t = t.replace(ch, " ")
    for ch in LINE_SEPARATORS:
        t = t.replace(ch, " ")
    t = (
        t.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )

    t = t.replace("\r\n", "\n").replace("\r", "\n")

    t = ALLOWED_TAG_RE.sub("", t)

    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.replace("\t", " ")

    t = re.sub(r" {2,}", " ", t)
    t = "\n".join(part.strip() for part in t.split("\n"))
    return t.strip()


def clean_snippet_list(snippets):
    """Làm sạch danh sách đoạn, giữ thứ tự và ranh giới (dùng \n\n khi ghép)."""
    if not isinstance(snippets, list):
        return []
    return [clean_text(s) for s in snippets if isinstance(s, str)]


def percentile_sorted(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (q / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def ham_stats(vals):
    """(trung vị, p90, max) - dùng cho báo cáo trước/sau."""
    s = sorted(vals)
    if not s:
        return (0.0, 0.0, 0)
    return (statistics.median(s), percentile_sorted(s, 90), s[-1])


def lam_sach_docs(input_path: Path, output_path: Path) -> dict:
    """Lam sach mot file docs_*.jsonl, ghi file moi, KHONG ghi de file goc.

    Ghi mọi trường cũ giữ nguyên + trường mới text_clean, n_chars_clean.
    Trả về thống kê để in báo cáo.
    """
    n_in = 0
    n_out = 0
    n_empty_after = 0
    n_residual_num = 0
    n_residual_named = 0
    below_min = 0
    doc_ids = []
    stats_before = []
    stats_after = []
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)
            t_before = obj.get("text_goc")
            t_clean = clean_text(t_before)
            if not t_clean.strip():
                n_empty_after += 1
            if RESIDUE_NUM_RE.search(t_clean):
                n_residual_num += 1
            if RESIDUE_NAMED_RE.search(t_clean):
                n_residual_named += 1
            stats_before.append(obj.get("n_chars"))
            stats_after.append(len(t_clean))
            doc_ids.append(obj.get("doc_id"))

            t = dict(obj)
            t["text_clean"] = t_clean
            t["n_chars_clean"] = len(t_clean)
            records.append(t)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_out = len(records)

    return {
        "n_in": n_in,
        "n_out": n_out,
        "n_empty_after": n_empty_after,
        "n_residual_num": n_residual_num,
        "n_residual_named": n_residual_named,
        "doc_ids": doc_ids,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "n_below_min_5000": sum(1 for v in stats_after if v < 5000),
    }


def ap_dung_lam_sach(
    processed_dir: Path,
    min_doc_chars: int,
    seed: int,
) -> dict:
    """Áp dụng clean_text cho docs_dev.jsonl và docs_test.jsonl -> *_clean.jsonl.

    KHÔNG ghi đè file gốc. Kiểm tra sau khi chạy, in thống kê và trả kết quả
    để báo cáo. Mọi số liệu đều đo từ lần chạy này.
    """
    ket_qua = {}
    doc_ids_cam = {}
    bo_goc = {}
    bo_sau = {}

    cap = [
        ("dev", processed_dir / "docs_dev.jsonl", processed_dir / "docs_dev_clean.jsonl"),
        ("test", processed_dir / "docs_test.jsonl", processed_dir / "docs_test_clean.jsonl"),
    ]
    for ten, in_path, out_path in cap:
        if not in_path.exists():
            raise FileNotFoundError(f"Thiếu file đầu vào: {in_path}")

        ids_goc = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids_goc.append(json.loads(line).get("doc_id"))

        r = lam_sach_docs(in_path, out_path)

        id_sau = r["doc_ids"]
        kiem_doc_id = (
            len(id_sau) == len(ids_goc)
            and id_sau == ids_goc
            and len(set(id_sau)) == len(id_sau)
        )
        r["kiem_doc_id"] = kiem_doc_id
        r["kiem_so_dong"] = r["n_out"] == r["n_in"]
        med, p90, mx = ham_stats(r["stats_before"])
        med2, p902, mx2 = ham_stats(r["stats_after"])
        r["before_stats"] = (med, p90, mx)
        r["after_stats"] = (med2, p902, mx2)
        ket_qua[ten] = r
        doc_ids_cam[ten] = id_sau

        goc_list = []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                goc_list.append(
                    {
                        "doc_id": obj["doc_id"],
                        "text_goc": obj["text_goc"],
                        "text_clean": obj["text_clean"],
                    }
                )
        bo_goc[ten] = goc_list

    rng = random.Random(seed)
    danh_sach = [("dev", i) for i in range(len(bo_goc["dev"]))] + [
        ("test", i) for i in range(len(bo_goc["test"]))
    ]
    kq_chon = rng.sample(danh_sach, 3)
    vi_du = []
    for ten, i in kq_chon:
        row = bo_goc[ten][i]
        vi_du.append(
            {
                "nguon": ten,
                "doc_id": row["doc_id"],
                "before300": row["text_goc"][:300],
                "after300": row["text_clean"][:300],
            }
        )
    return {"ket_qua": ket_qua, "vi_du": vi_du}


def chon_reference_answer(answers_clean: list) -> dict:
    """Chọn câu trả lời tham chiếu tốt nhất từ danh sách answers_clean.

    Tiêu chí xếp hạng ưu tiên:
    1. Không rỗng (len > 0).
    2. Không thuộc nhóm 'don't know / not sure / no idea'.
    3. Có độ dài >= 5 ký tự (tránh câu cụt 'Yes.' nếu có câu giải thích tốt hơn
       cùng mức helpful, nhưng vẫn ưu tiên điểm helpful trước).
    4. Điểm helpful ròng (2 * upvotes - total_votes) giảm dần.
    5. Số lượt upvotes giảm dần.
    6. Thứ tự xuất hiện gốc tăng dần (giữ ưu tiên đầu danh sách của AmazonQA).
    """
    if not isinstance(answers_clean, list) or not answers_clean:
        return {"reference_answer": "", "reference_helpful": [0, 0]}

    ung_vien = []
    for idx, ans in enumerate(answers_clean):
        if not isinstance(ans, dict):
            continue
        text_c = ans.get("answer_text_clean", "")
        if not isinstance(text_c, str):
            text_c = clean_text(text_c)
        helpful = ans.get("helpful", [0, 0])
        if (
            isinstance(helpful, list)
            and len(helpful) == 2
            and isinstance(helpful[0], (int, float))
            and isinstance(helpful[1], (int, float))
        ):
            up = int(helpful[0])
            tot = int(helpful[1])
        else:
            up, tot = 0, 0
            helpful = [0, 0]

        net_helpful = 2 * up - tot
        dont_know = bool(ans.get("is_dont_know", is_dont_know_answer(text_c)))
        non_empty = 1 if len(text_c.strip()) > 0 else 0
        not_dk = 0 if dont_know else 1

        ung_vien.append(
            (
                (non_empty, not_dk, net_helpful, up, -idx),
                text_c,
                [up, tot],
            )
        )

    if not ung_vien:
        return {"reference_answer": "", "reference_helpful": [0, 0]}

    ung_vien.sort(key=lambda item: item[0], reverse=True)
    _, best_text, best_helpful = ung_vien[0]
    return {"reference_answer": best_text, "reference_helpful": best_helpful}


def lam_sach_questions(input_path: Path, output_path: Path) -> dict:
    """Làm sạch file questions_*.jsonl và ghi ra file mới questions_*_clean.jsonl.

    Giữ nguyên mọi trường cũ, bổ sung:
    - question_text_clean, n_chars_question_clean
    - answers_clean (danh sách dict có thêm answer_text_clean, is_dont_know)
    - reference_answer, reference_helpful
    - review_snippets_clean
    """
    n_in = 0
    n_out = 0
    n_corrupt = 0
    n_q_changed = 0
    n_a_changed = 0
    n_s_changed = 0
    n_empty_q_after = 0
    n_empty_ref_after = 0
    qids = []
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                n_corrupt += 1
                continue
            if not isinstance(obj, dict):
                n_corrupt += 1
                continue

            n_in += 1
            q_raw = obj.get("question_text", "")
            q_clean = clean_text(q_raw)
            if q_clean != q_raw:
                n_q_changed += 1
            if not q_clean.strip():
                n_empty_q_after += 1

            answers_raw = obj.get("answers", [])
            answers_clean = []
            if isinstance(answers_raw, list):
                for a in answers_raw:
                    if not isinstance(a, dict):
                        continue
                    at_raw = a.get("answerText", "")
                    at_clean = clean_text(at_raw)
                    if at_clean != at_raw:
                        n_a_changed += 1
                    a_new = dict(a)
                    a_new["answer_text_clean"] = at_clean
                    a_new["is_dont_know"] = is_dont_know_answer(at_clean)
                    answers_clean.append(a_new)

            ref_info = chon_reference_answer(answers_clean)
            if not ref_info["reference_answer"].strip():
                n_empty_ref_after += 1

            snippets_raw = obj.get("review_snippets_cau_hoi", [])
            snippets_clean = clean_snippet_list(snippets_raw)
            if isinstance(snippets_raw, list):
                for s_raw, s_clean in zip(snippets_raw, snippets_clean):
                    if s_raw != s_clean:
                        n_s_changed += 1

            rec = dict(obj)
            rec["question_text_clean"] = q_clean
            rec["n_chars_question_clean"] = len(q_clean)
            rec["answers_clean"] = answers_clean
            rec["reference_answer"] = ref_info["reference_answer"]
            rec["reference_helpful"] = ref_info["reference_helpful"]
            rec["review_snippets_clean"] = snippets_clean
            records.append(rec)
            qids.append(obj.get("qid_moi"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_out = len(records)

    return {
        "n_in": n_in,
        "n_out": n_out,
        "n_corrupt": n_corrupt,
        "n_q_changed": n_q_changed,
        "n_a_changed": n_a_changed,
        "n_s_changed": n_s_changed,
        "n_empty_q_after": n_empty_q_after,
        "n_empty_ref_after": n_empty_ref_after,
        "kiem_qid_duy_nhat": len(qids) == len(set(qids)),
        "kiem_so_dong": n_in == n_out,
    }


def ap_dung_lam_sach_questions(processed_dir: Path) -> dict:
    """Áp dụng lam_sach_questions cho questions_dev.jsonl và questions_test.jsonl."""
    cap = [
        (
            "dev",
            processed_dir / "questions_dev.jsonl",
            processed_dir / "questions_dev_clean.jsonl",
        ),
        (
            "test",
            processed_dir / "questions_test.jsonl",
            processed_dir / "questions_test_clean.jsonl",
        ),
    ]
    ket_qua = {}
    for ten, in_path, out_path in cap:
        if not in_path.exists():
            raise FileNotFoundError(f"Thiếu file câu hỏi đầu vào: {in_path}")
        ket_qua[ten] = lam_sach_questions(in_path, out_path)
    return ket_qua
