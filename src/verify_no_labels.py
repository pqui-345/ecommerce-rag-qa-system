"""Kiểm tra nhãn no/partial: tìm đoạn ứng viên trong TOÀN TÀI LIỆC bằng BM25.

Đầu vào:
- File nhãn (mặc định reports/ung_vien_kiem_tay.csv, cột nhan_tay: no/partial/yes).
- data/processed/questions_dev.jsonl (câu hỏi, nối bằng doc_id).
- data/processed/docs_dev_clean.jsonl (text_clean đã làm sạch).

Cách làm:
1. Tách text_clean của tài liệu theo ranh giới "\\n\\n".
2. Loại các đoạn trùng với 10 đoạn gốc của chính câu hỏi (so khớp sau chuẩn hóa
   khoảng trắng, không phân biệt hoa/thường).
3. Xếp hạng các đoạn còn lại bằng BM25 (tự cài đặt, k1=1.5, b=0.75) với truy vấn
   là questionText đã tách từ và bỏ stopword tiếng Anh. Không dùng embedding,
   không gọi LLM.
4. Ghi CSV: qid_moi, nhan_cuoi, question_text, so_doan_con_lai, top1..top3
   (điểm BM25 + 250 ký tự đầu đoạn), cot co_trong_tai_lieu để trống.
5. Sắp xếp theo điểm top1 giảm dần; KHÔNG đặt ngưỡng loại câu.
"""

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from src.clean import clean_text

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "does", "didn", "won", "don", "ain", "ve", "ll", "re", "d",
    "m", "o", "y", "s", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn",
})

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tach_doan(text) -> list:
    """Tách tài liệu thành các đoạn theo ranh giới "\\n\\n", bỏ đoạn rỗng."""
    if not isinstance(text, str):
        return []
    return [seg for seg in text.split("\n\n") if seg.strip()]


def chuan_hoa(s) -> str:
    """Chuẩn hóa để so khớp trùng: strip + lower + gộp khoảng trắng."""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.strip().lower().split())


def loai_doan_goc(doan_doc, doan_cau_hoi):
    """Loại các đoạn của tài liệu trùng với đoạn gốc của câu hỏi.

    So khớp trên bản chuan_hoa; giữ thứ tự gặp của các đoạn còn lại.
    Trả về danh sách đoạn gốc (không sửa đổi nội dung).
    """
    bo = {chuan_hoa(x) for x in (doan_cau_hoi or []) if isinstance(x, str)}
    return [seg for seg in (doan_doc or []) if chuan_hoa(seg) not in bo]


def doan_cau_hoi_da_lam_sach(doan_cau_hoi):
    """Áp clean_text cho các đoạn gốc của câu hỏi TRƯỚC khi so khớp.

    Lý do: đoạn gốc lấy từ review_snippets_cau_hoi là bản CHƯA làm sạch
    (có thể chứa &#34; v.v.), còn tài liệu là text_clean đã giải mã; nếu
    không làm sạch trước khi so khớp thì đoạn gốc trùng nội dung sẽ KHÔNG
    bị loại khỏi ứng viên.
    """
    return [
        clean_text(x) for x in (doan_cau_hoi or []) if isinstance(x, str)
    ]


def tokenize(s) -> list:
    """Tách từ: chữ số/chữ cái thường, bỏ mọi ký tự khác."""
    if not isinstance(s, str):
        s = str(s)
    return TOKEN_RE.findall(s.lower())


def bo_stopword(tokens) -> list:
    """Bỏ stopword tiếng Anh khỏi danh sách token (giữ số)."""
    return [t for t in (tokens or []) if t not in STOPWORDS]


class BM25TinhDiem:
    """BM25 (Okapi) tự cài đặt. k1=1.5, b=0.75 (tham số mặc định phổ biến)."""

    def __init__(self, doan_tokens_list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokens_list = [list(t) for t in doan_tokens_list]
        self.N = len(self.tokens_list)
        self.avgdl = (
            sum(len(t) for t in self.tokens_list) / self.N if self.N else 0.0
        )
        self.tf_list = [Counter(t) for t in self.tokens_list]
        self.df = Counter()
        for tf in self.tf_list:
            self.df.update(tf.keys())
        self.idf = {
            w: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for w, df in self.df.items()
        }

    def diem(self, q_tokens, idx: int) -> float:
        """Điểm BM25 của đoạn idx với truy vấn q_tokens (đã bỏ stopword)."""
        if self.N == 0 or self.avgdl == 0 or not q_tokens:
            return 0.0
        tf = self.tf_list[idx]
        dl = len(self.tokens_list[idx])
        diem = 0.0
        for w in q_tokens:  # lặp cả từ lặp lại -> như rank_bm25
            if w not in tf:
                continue
            f = tf[w]
            d = self.idf.get(w, 0.0)
            diem += d * f * (self.k1 + 1.0) / (
                f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            )
        return diem


def xu_hao_top(remaining_segments, q_tokens, top_n: int = 3):
    """Xếp hạng các đoạn còn lại theo BM25, trả về top_n (điểm, đoạn)."""
    corpus = [tokenize(seg) for seg in remaining_segments]
    bm = BM25TinhDiem(corpus)
    ket_qua = sorted(
        ((bm.diem(q_tokens, i), seg) for i, seg in enumerate(remaining_segments)),
        key=lambda x: -x[0],
    )
    return ket_qua[:top_n]


def kiem_tra_no(
    nhan_csv: Path,
    processed_dir: Path,
    out_csv: Path,
    cot_nhan: str = "nhan_tay",
    cac_nhan_xet: tuple = ("no", "partial"),
    so_top: int = 3,
    do_dai_doan: int = 250,
) -> dict:
    """Chạy toàn bộ quy trình, ghi CSV và trả thống kê."""
    if not nhan_csv.exists():
        raise FileNotFoundError(f"Thiếu file nhãn: {nhan_csv}")
    with open(nhan_csv, "r", encoding="utf-8-sig") as f:
        nhan = {r["qid_moi"]: (r[cot_nhan] or "").strip().lower() for r in csv.DictReader(f)}

    questions = {}
    with open(processed_dir / "questions_dev.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            questions[q["qid_moi"]] = q

    docs_clean = {}
    with open(processed_dir / "docs_dev_clean.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs_clean[d["doc_id"]] = d.get("text_clean", "")

    thieu_qid = sorted(set(nhan) - set(questions))
    if thieu_qid:
        print(f"CANH BAO: {len(thieu_qid)} qid_moi trong file nhan khong co trong questions_dev.jsonl: {thieu_qid[:5]}")

    ket_qua = []
    da_bo_qua = Counter()
    for qid_moi, ncuoi in sorted(nhan.items()):
        if ncuoi not in cac_nhan_xet:
            da_bo_qua[ncuoi] += 1
            continue
        q = questions.get(qid_moi)
        if q is None:
            da_bo_qua["thieu_trong_questions"] += 1
            continue
        text_clean = docs_clean.get(q["doc_id"], "")
        doan_doc = tach_doan(text_clean)
        doan_cau = doan_cau_hoi_da_lam_sach(q.get("review_snippets_cau_hoi", []) or [])
        con_lai = loai_doan_goc(doan_doc, doan_cau)
        q_tokens = bo_stopword(tokenize(q.get("question_text", "")))
        tops = xu_hao_top(con_lai, q_tokens, top_n=so_top)
        row = {
            "qid_moi": qid_moi,
            "nhan_cuoi": ncuoi,
            "question_text": q.get("question_text", ""),
            "so_doan_con_lai": len(con_lai),
            "co_trong_tai_lieu": "",
        }
        for i in range(so_top):
            if i < len(tops):
                diem, seg = tops[i]
                row[f"top{i + 1}_diem"] = round(float(diem), 4)
                row[f"top{i + 1}_doan"] = (seg or "")[:do_dai_doan]
            else:
                row[f"top{i + 1}_diem"] = ""
                row[f"top{i + 1}_doan"] = ""
        row["_top1_float"] = float(tops[0][0]) if tops else 0.0
        ket_qua.append(row)

    ket_qua.sort(key=lambda r: -r["_top1_float"])
    for r in ket_qua:
        r.pop("_top1_float")

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "qid_moi", "nhan_cuoi", "question_text", "so_doan_con_lai",
            "top1_diem", "top1_doan", "top2_diem", "top2_doan",
            "top3_diem", "top3_doan", "co_trong_tai_lieu",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ket_qua)

    phan_bo = Counter(r["nhan_cuoi"] for r in ket_qua)
    top1 = [float(r["top1_diem"]) for r in ket_qua if r["top1_diem"] != ""]
    top1_sorted = sorted(top1)
    n = len(top1_sorted)
    k = 0.9 * (n - 1)
    f_idx, c_idx = int(k), min(int(k) + 1, n - 1)
    p90 = top1_sorted[f_idx] if f_idx == c_idx else (
        top1_sorted[f_idx] * (c_idx - k) + top1_sorted[c_idx] * (k - f_idx)
    )
    thong_ke = {
        "n_xu_ly": len(ket_qua),
        "phan_bo": dict(phan_bo),
        "da_bo_qua": dict(da_bo_qua),
        "top1_min": top1_sorted[0] if n else 0.0,
        "top1_median": statistics.median(top1_sorted) if n else 0.0,
        "top1_p90": p90 if n else 0.0,
        "top1_max": top1_sorted[-1] if n else 0.0,
        "n_it_hon_3_doan": sum(1 for r in ket_qua if r["so_doan_con_lai"] < 3),
        "n_top1_bang_0": sum(1 for r in ket_qua if r["top1_diem"] == 0 or r["top1_diem"] == ""),
    }
    return {"ket_qua": ket_qua, "thong_ke": thong_ke, "out_csv": out_csv}
