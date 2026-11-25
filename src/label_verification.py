"""Kiểm định độ tin cậy của nhãn tự động (is_answerable_auto) so với nhãn người
trên 120 câu ứng viên (ung_vien_kiem_tay.csv) và 93 câu rà soát toàn tài liệu
(kiem_tra_93_cau_da_ra_soat_chot.csv), đồng thời đóng gói tập đánh giá có nhãn
kiểm tay vào data/eval/eval_120_dev_labeled.jsonl.

Quy ước theo AGENTS.md:
- Đường dẫn tính bằng pathlib từ vị trí file.
- Không ghi đè file dữ liệu gốc; ghi file mới.
- Có hàm tính Cohen's kappa, ma trận nhầm lẫn, Precision/Recall/F1.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent


def _rel_or_str(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(p)


def tinh_cohens_kappa(y_a: Sequence[int], y_b: Sequence[int]) -> dict:
    """Tính ma trận nhầm lẫn 2x2, Accuracy, Precision, Recall, F1 và Cohen's kappa.

    y_a: nhãn dự đoán / tự động (0 hoặc 1).
    y_b: nhãn chuẩn của người kiểm định (0 hoặc 1).
    """
    if len(y_a) != len(y_b):
        raise ValueError("Hai dãy nhãn phải có cùng độ dài.")
    n = len(y_a)
    if n == 0:
        return {
            "n": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "po": 0.0,
            "pe": 0.0,
            "kappa": 0.0,
        }

    tp = sum(1 for a, b in zip(y_a, y_b) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_a, y_b) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(y_a, y_b) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(y_a, y_b) if a == 0 and b == 0)

    po = (tp + tn) / n
    p_a1 = (tp + fp) / n
    p_a0 = (fn + tn) / n
    p_b1 = (tp + fn) / n
    p_b0 = (fp + tn) / n
    pe = p_a1 * p_b1 + p_a0 * p_b0

    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(po, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "po": round(po, 4),
        "pe": round(pe, 4),
        "kappa": round(kappa, 4),
    }


def xac_dinh_nhan_chot(nhan_tay_10: str, co_trong_doc: str) -> Tuple[str, str, int]:
    """Hợp nhất nhãn kiểm tay trên 10 đoạn gốc và nhãn rà soát trên toàn tài liệu.

    Trả về (nhan_chot_3_muc, nhan_chot_chi_tiet, co_thong_tin_toan_doc_int):
    - nhan_chot_3_muc:
        * 'yes' (trả lời đầy đủ ngay từ 10 đoạn gốc)
        * 'partial' (có thông tin liên quan một phần trong 10 đoạn hoặc các đoạn còn lại)
        * 'no' (hoàn toàn không có bằng chứng trong toàn bộ tài liệu asin)
    - nhan_chot_chi_tiet:
        * 'answerable_full'
        * 'partial_expanded'
        * 'partial_10_only'
        * 'recovered_in_full_doc'
        * 'unanswerable_in_doc'
    - co_thong_tin_toan_doc_int: 1 nếu có thông tin trong tài liệu, 0 nếu hoàn toàn không.
    """
    nt = (nhan_tay_10 or "").strip().lower()
    co = (co_trong_doc or "").strip().lower()

    if nt == "yes":
        return ("yes", "answerable_full", 1)
    if nt == "partial":
        if co == "co":
            return ("partial", "partial_expanded", 1)
        return ("partial", "partial_10_only", 1)
    if nt == "no":
        if co == "co":
            return ("partial", "recovered_in_full_doc", 1)
        return ("no", "unanswerable_in_doc", 0)
    raise ValueError(f"Nhãn tay không hợp lệ: {nhan_tay_10!r}")


def tong_hop_kiem_dinh_nhan(
    ung_vien_csv: Path,
    ra_soat_93_csv: Path,
    questions_clean_jsonl: Path,
    eval_out_jsonl: Path,
    report_out_md: Path,
) -> dict:
    """Đọc 2 file CSV nhãn tay + questions_dev_clean.jsonl, xuất eval set & báo cáo."""
    if not ung_vien_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {ung_vien_csv}")
    if not ra_soat_93_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {ra_soat_93_csv}")
    if not questions_clean_jsonl.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {questions_clean_jsonl}")

    with open(ung_vien_csv, "r", encoding="utf-8-sig") as f:
        uv_rows = [r for r in csv.DictReader(f) if r.get("qid_moi")]

    with open(ra_soat_93_csv, "r", encoding="utf-8-sig") as f:
        rs_map: Dict[str, dict] = {
            r["qid_moi"]: r for r in csv.DictReader(f) if r.get("qid_moi")
        }

    q_map: Dict[str, dict] = {}
    with open(questions_clean_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("qid_moi"):
                q_map[obj["qid_moi"]] = obj

    eval_records: List[dict] = []
    y_auto: List[int] = []
    y_strict_10: List[int] = []
    y_lenient_10: List[int] = []
    y_full_doc: List[int] = []

    dem_3_muc = Counter()
    dem_chi_tiet = Counter()
    dem_theo_ia_va_chi_tiet = Counter()
    dem_theo_qtype_va_3_muc = Counter()

    for r in uv_rows:
        qid = r["qid_moi"]
        if qid not in q_map:
            raise KeyError(f"qid_moi {qid} không tồn tại trong {questions_clean_jsonl}")
        q = q_map[qid]
        nt10 = (r.get("nhan_tay") or "").strip().lower()
        ia_auto = int(r.get("is_answerable_auto", q.get("is_answerable_auto", 0)))

        if nt10 == "yes":
            co_doc = "co"
            rs_info = {}
        else:
            if qid not in rs_map:
                raise KeyError(
                    f"qid_moi {qid} (nhan_tay={nt10}) thiếu trong {ra_soat_93_csv}"
                )
            rs_info = rs_map[qid]
            co_doc = (rs_info.get("co_trong_tai_lieu") or "").strip().lower()

        nhan_3_muc, nhan_chi_tiet, co_toan_doc_int = xac_dinh_nhan_chot(nt10, co_doc)

        y_auto.append(ia_auto)
        y_strict_10.append(1 if nt10 == "yes" else 0)
        y_lenient_10.append(1 if nt10 in ("yes", "partial") else 0)
        y_full_doc.append(co_toan_doc_int)

        dem_3_muc[nhan_3_muc] += 1
        dem_chi_tiet[nhan_chi_tiet] += 1
        dem_theo_ia_va_chi_tiet[(ia_auto, nhan_chi_tiet)] += 1
        dem_theo_qtype_va_3_muc[(q["question_type"], nhan_3_muc)] += 1

        vi_tri_raw = (rs_info.get("vi_tri_review_trong_doc") or "").strip()
        vi_tri_int = int(vi_tri_raw) if vi_tri_raw.isdigit() else None

        eval_rec = {
            "qid_moi": qid,
            "asin": q["asin"],
            "doc_id": q["doc_id"],
            "question_text": q["question_text"],
            "question_text_clean": q["question_text_clean"],
            "question_type": q["question_type"],
            "is_answerable_auto": ia_auto,
            "nhan_tay_10_snippets": nt10,
            "co_trong_toan_tai_lieu": co_doc,
            "nhan_chot_3_muc": nhan_3_muc,
            "nhan_chot_chi_tiet": nhan_chi_tiet,
            "co_thong_tin_trong_doc": bool(co_toan_doc_int),
            "ket_luan_muc_do": rs_info.get(
                "ket_luan_muc_do",
                "Trả lời đầy đủ ngay trong 10 đoạn review gốc" if nt10 == "yes" else "",
            ),
            "bang_chung_ra_soat": rs_info.get("bang_chung_ra_soat", ""),
            "vi_tri_review_trong_doc": vi_tri_int,
            "ly_do_chot": rs_info.get("ly_do_chot", r.get("ghi_chu", "")),
            "ghi_chu_kiem_tay_10_doan": r.get("ghi_chu", ""),
            "reference_answer": q["reference_answer"],
            "reference_helpful": q["reference_helpful"],
            "answers_clean": q["answers_clean"],
            "review_snippets_clean": q["review_snippets_clean"],
            "uids": q["uids"],
        }
        eval_records.append(eval_rec)

    eval_out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out_jsonl, "w", encoding="utf-8") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    m_strict = tinh_cohens_kappa(y_auto, y_strict_10)
    m_lenient = tinh_cohens_kappa(y_auto, y_lenient_10)
    m_fulldoc = tinh_cohens_kappa(y_auto, y_full_doc)

    lines = [
        "# BÁO CÁO KIỂM ĐỊNH NHÃN TỰ ĐỘNG VÀ ĐÓNG GÓI TẬP ĐÁNH GIÁ (120 CÂU DEV)",
        "",
        "## 1. Nguồn dữ liệu đối chiếu",
        f"- File kiểm tay 10 đoạn gốc: `{_rel_or_str(ung_vien_csv)}` (**{len(uv_rows)}** câu hỏi, chia đều 4 tổ hợp `question_type` × `is_answerable_auto`, mỗi tổ hợp 30 câu).",
        f"- File rà soát toàn tài liệu (BM25 + đọc kiểm tra các đoạn còn lại): `{_rel_or_str(ra_soat_93_csv)}` (**{len(rs_map)}** câu thuộc nhóm `no`/`partial`).",
        f"- File câu hỏi đã làm sạch: `{_rel_or_str(questions_clean_jsonl)}`.",
        f"- File tập đánh giá xuất ra: `{_rel_or_str(eval_out_jsonl)}` (**{len(eval_records)}** dòng JSONL).",
        "",
        "## 2. Kết quả phân tầng nhãn sau khi rà soát toàn tài liệu (`doc_id`)",
        "",
        "| Nhãn chốt 3 mức (`nhan_chot_3_muc`) | Nhãn chi tiết (`nhan_chot_chi_tiet`) | Ý nghĩa thực nghiệm | Số câu | Tỷ lệ (%) |",
        "|---|---|---|---:|---:|",
        f"| `yes` | `answerable_full` | 10 đoạn review gốc đã đủ trả lời trọn vẹn câu hỏi | {dem_chi_tiet['answerable_full']} | {dem_chi_tiet['answerable_full']*100/len(eval_records):.2f}% |",
        f"| `partial` | `partial_expanded` | 10 đoạn gốc trả lời một phần + các đoạn khác trong tài liệu có thêm bằng chứng | {dem_chi_tiet['partial_expanded']} | {dem_chi_tiet['partial_expanded']*100/len(eval_records):.2f}% |",
        f"| `partial` | `partial_10_only` | Chỉ có thông tin một phần ở 10 đoạn gốc; phần còn lại của tài liệu không có thêm | {dem_chi_tiet['partial_10_only']} | {dem_chi_tiet['partial_10_only']*100/len(eval_records):.2f}% |",
        f"| `partial` | `recovered_in_full_doc` | 10 đoạn gốc KHÔNG có (`no`), nhưng tìm thấy thông tin liên quan trong toàn tài liệu `asin` | {dem_chi_tiet['recovered_in_full_doc']} | {dem_chi_tiet['recovered_in_full_doc']*100/len(eval_records):.2f}% |",
        f"| `no` | `unanswerable_in_doc` | Hoàn toàn KHÔNG có bằng chứng trong toàn bộ tài liệu (câu bẫy chống ảo giác tự nhiên) | {dem_chi_tiet['unanswerable_in_doc']} | {dem_chi_tiet['unanswerable_in_doc']*100/len(eval_records):.2f}% |",
        f"| **TỔNG** | | | **{len(eval_records)}** | **100.00%** |",
        "",
        "### Tổng hợp theo 3 nhóm lớn trên toàn tài liệu:",
        f"- **Trả lời đầy đủ (`yes`):** {dem_3_muc['yes']} câu ({dem_3_muc['yes']*100/len(eval_records):.2f}%)",
        f"- **Có thông tin một phần / liên quan (`partial`):** {dem_3_muc['partial']} câu ({dem_3_muc['partial']*100/len(eval_records):.2f}%)",
        f"- **Không thể trả lời từ tài liệu (`no` / `unanswerable_in_doc`):** {dem_3_muc['no']} câu ({dem_3_muc['no']*100/len(eval_records):.2f}%)",
        "",
        "## 3. Bảng chéo giữa nhãn tự động `is_answerable_auto` và nhãn chi tiết của người",
        "",
        "| `nhan_chot_chi_tiet` | `is_answerable_auto = 0` (n=60) | `is_answerable_auto = 1` (n=60) | Tổng (n=120) |",
        "|---|---:|---:|---:|",
    ]

    for k in (
        "answerable_full",
        "partial_expanded",
        "partial_10_only",
        "recovered_in_full_doc",
        "unanswerable_in_doc",
    ):
        c0 = dem_theo_ia_va_chi_tiet[(0, k)]
        c1 = dem_theo_ia_va_chi_tiet[(1, k)]
        lines.append(f"| `{k}` | {c0} | {c1} | {c0 + c1} |")

    lines.extend(
        [
            "",
            "## 4. Định lượng độ tin cậy của nhãn tự động `is_answerable_auto` (Cohen's kappa)",
            "",
            "Đối chiếu nhãn máy `is_answerable_auto` (0/1) với nhãn người kiểm định dưới 3 cách quy chiếu nhị phân:",
            "1. **Quy chiếu A (Khắt khe - 10 đoạn gốc):** Chỉ coi `yes` là 1 (27 câu); `partial` và `no` là 0 (93 câu).",
            "2. **Quy chiếu B (Bao dung - 10 đoạn gốc):** Coi `yes + partial` là 1 (72 câu); `no` là 0 (48 câu).",
            "3. **Quy chiếu C (Toàn tài liệu `doc_id`):** Coi mọi câu có thông tin trong tài liệu (`yes + partial + recovered_in_full_doc` = 91 câu) là 1; hoàn toàn không có (`unanswerable_in_doc` = 29 câu) là 0.",
            "",
            "| Quy chiếu | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | $P_o$ | $P_e$ | **Cohen's $\\kappa$** |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| **A. Khắt khe (10 đoạn)** | {m_strict['tp']} | {m_strict['fp']} | {m_strict['fn']} | {m_strict['tn']} | {m_strict['accuracy']:.4f} | {m_strict['precision']:.4f} | {m_strict['recall']:.4f} | {m_strict['f1']:.4f} | {m_strict['po']:.4f} | {m_strict['pe']:.4f} | **{m_strict['kappa']:.4f}** |",
            f"| **B. Bao dung (10 đoạn)** | {m_lenient['tp']} | {m_lenient['fp']} | {m_lenient['fn']} | {m_lenient['tn']} | {m_lenient['accuracy']:.4f} | {m_lenient['precision']:.4f} | {m_lenient['recall']:.4f} | {m_lenient['f1']:.4f} | {m_lenient['po']:.4f} | {m_lenient['pe']:.4f} | **{m_lenient['kappa']:.4f}** |",
            f"| **C. Toàn tài liệu (`doc_id`)** | {m_fulldoc['tp']} | {m_fulldoc['fp']} | {m_fulldoc['fn']} | {m_fulldoc['tn']} | {m_fulldoc['accuracy']:.4f} | {m_fulldoc['precision']:.4f} | {m_fulldoc['recall']:.4f} | {m_fulldoc['f1']:.4f} | {m_fulldoc['po']:.4f} | {m_fulldoc['pe']:.4f} | **{m_fulldoc['kappa']:.4f}** |",
            "",
            "### Nhận xét khoa học từ số liệu thực nghiệm:",
            f"- Hệ số **Cohen's $\\kappa$** dao động từ **{m_strict['kappa']:.4f}** đến **{m_lenient['kappa']:.4f}** (mức đồng thuận *yếu / slight-to-fair agreement* theo thang Landis & Koch). Điều này chứng minh bằng số liệu rằng nhãn `is_answerable` tự sinh trong bộ AmazonQA gốc bị nhiễu rất mạnh và **không thể dùng trực tiếp làm ground-truth nếu không qua bước kiểm tay**.",
            f"- Nhờ gộp toàn bộ đoạn review theo `asin` thành tài liệu hoàn chỉnh, có **19/48 câu (`39.58%`)** vốn bị thiếu bằng chứng ở 10 đoạn top IR ban đầu (`no`) đã tìm lại được thông tin liên quan trong toàn tài liệu (`recovered_in_full_doc`).",
            f"- Tập 120 câu sau kiểm định cung cấp đồng thời cả 3 nhóm câu hỏi chuẩn cho thí nghiệm RAG: **27 câu trả lời đầy đủ**, **64 câu có thông tin một phần/đa khía cạnh**, và **29 câu hoàn toàn không có thông tin trong tài liệu (`unanswerable_in_doc`)** để định lượng khả năng chống ảo giác (từ chối trả lời khi ngữ cảnh không đủ).",
            "",
        ]
    )

    report_out_md.parent.mkdir(parents=True, exist_ok=True)
    report_out_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_eval": len(eval_records),
        "dem_3_muc": dict(dem_3_muc),
        "dem_chi_tiet": dict(dem_chi_tiet),
        "metrics_strict_10": m_strict,
        "metrics_lenient_10": m_lenient,
        "metrics_full_doc": m_fulldoc,
        "eval_out_jsonl": str(eval_out_jsonl),
        "report_out_md": str(report_out_md),
    }
