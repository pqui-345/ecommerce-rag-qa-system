"""Dựng bộ làm việc nhỏ từ amazonqa_merged_v2.jsonl (đọc streaming, không nạp cả file vào RAM)."""

import csv
import hashlib
import json
import logging
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - pyyaml là phụ thuộc bắt buộc của dự án
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config(config_path: Path) -> dict:
    """Đọc tệp cấu hình config.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    if yaml is None:
        raise RuntimeError("Thiếu thư viện pyyaml (cần thiết để đọc config.yaml).")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def norm_text(s) -> str:
    """Chuẩn hóa tạm để so khớp: strip + lower + gộp khoảng trắng (không sửa file gốc)."""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.strip().lower().split())


def segment_key(s) -> str:
    """Hash sha1 của đoạn review_snippets sau khi chuẩn hóa (chỉ lưu hash trong RAM)."""
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()


DONT_KNOW_RE = re.compile(
    r"(don['’]t|dont|do not)\s+know|idk|not\s+sure|no\s+idea",
    re.IGNORECASE,
)


def is_dont_know(answer_text) -> bool:
    if not isinstance(answer_text, str):
        return False
    return bool(DONT_KNOW_RE.search(answer_text))


def percentile(sorted_vals, q: float) -> float:
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


def stats_of(vals):
    """Trả về (min, trung vị, p90, max)."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return (0, 0.0, 0.0, 0)
    return (s[0], float(statistics.median(s)), percentile(s, 90), s[-1])


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bo_lam_viec")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def write_jsonl(path: Path, records):
    """Ghi danh sách đối tượng ra JSONL (UTF-8, ensure_ascii=False)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def bo_lam_viec(
    merged_path: Path,
    processed_dir: Path,
    report_dir: Path,
    log_path: Path,
    category: str,
    min_doc_chars: int,
    n_asin_dev: int,
    n_asin_test: int,
    seed: int,
    min_question_chars: int,
    n_kiem_tay: int,
) -> dict:
    logger = setup_logging(log_path)
    rng = random.Random(seed)

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    asin_val = set()
    asin_test = set()
    test_lines_by_asin = defaultdict(int)

    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            source = obj.get("nguon_goc")
            asin = obj.get("asin", "")
            if source == "val":
                asin_val.add(asin)
            elif source == "test":
                asin_test.add(asin)
                test_lines_by_asin[asin] += 1

    asin_chung = asin_val & asin_test
    n_asin_chung = len(asin_chung)
    n_test_lines_loai = sum(test_lines_by_asin[a] for a in asin_chung)
    logger.info(
        f"Bước 1: asin chung val & test = {n_asin_chung}; "
        f"số dòng test bị loại = {n_test_lines_loai}"
    )
    if n_asin_chung != 15 or n_test_lines_loai != 115:
        logger.warning(
            f"CẢNH BÁO: kỳ vọng 15 asin / 115 dòng, thực tế {n_asin_chung} / {n_test_lines_loai}"
        )

    docs = {}
    seen_segment_keys = defaultdict(set)

    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            source = obj.get("nguon_goc")
            asin = obj.get("asin", "")
            cat = obj.get("category", "")
            if cat != category:
                continue
            if source == "test" and asin in asin_chung:
                continue

            if asin not in docs:
                docs[asin] = {
                    "asin": asin,
                    "category": cat,
                    "nguon_goc": source,
                    "snippets": [],
                    "n_questions": 0,
                }

            doc = docs[asin]
            snippets = obj.get("review_snippets", [])
            if isinstance(snippets, list):
                for seg in snippets:
                    k = segment_key(seg)
                    if k not in seen_segment_keys[asin]:
                        seen_segment_keys[asin].add(k)
                        doc["snippets"].append(seg)
            doc["n_questions"] += 1

    for asin, doc in docs.items():
        doc["text_goc"] = "\n\n".join(doc["snippets"])
        doc["n_snippets"] = len(doc["snippets"])
        doc["n_chars"] = len(doc["text_goc"])
        doc["doc_id"] = f"doc-{asin}"

    docs_dev_candidates = []
    docs_test_candidates = []
    for asin, doc in docs.items():
        if doc["n_chars"] < min_doc_chars:
            continue
        if doc["nguon_goc"] == "val":
            docs_dev_candidates.append(asin)
        elif doc["nguon_goc"] == "test":
            docs_test_candidates.append(asin)

    docs_dev_candidates.sort()
    docs_test_candidates.sort()

    if len(docs_dev_candidates) < n_asin_dev:
        logger.warning(
            f"Không đủ asin dev đủ điều kiện: có {len(docs_dev_candidates)}, cần {n_asin_dev}"
        )
    if len(docs_test_candidates) < n_asin_test:
        logger.warning(
            f"Không đủ asin test đủ điều kiện: có {len(docs_test_candidates)}, cần {n_asin_test}"
        )

    asin_dev = rng.sample(
        docs_dev_candidates, min(n_asin_dev, len(docs_dev_candidates))
    )
    asin_test = rng.sample(
        docs_test_candidates, min(n_asin_test, len(docs_test_candidates))
    )
    asin_dev_set = set(asin_dev)
    asin_test_set = set(asin_test)

    questions = {}
    n_short_question = 0

    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            source = obj.get("nguon_goc")
            asin = obj.get("asin", "")
            if asin not in asin_dev_set and asin not in asin_test_set:
                continue
            cat = obj.get("category", "")
            if cat != category:
                continue

            qtext = obj.get("questionText", "")
            if not isinstance(qtext, str):
                qtext = str(qtext)
            if len(qtext) < min_question_chars:
                n_short_question += 1
                continue

            qnorm = norm_text(qtext)
            qid = obj.get("qid")
            uid = f"{source}-{qid}"

            key = (asin, qnorm)
            if key not in questions:
                questions[key] = {
                    "asin": asin,
                    "question_text": qtext,
                    "question_type": obj.get("questionType", ""),
                    "first_is_answerable": obj.get("is_answerable"),
                    "first_review_snippets": obj.get(
                        "review_snippets", []
                    )
                    if isinstance(obj.get("review_snippets"), list)
                    else [],
                    "answers": [],
                    "uids": [],
                    "n_dong_gop": 0,
                    "is_answerable_mismatch": 0,
                    "review_snippets_mismatch_count": 0,
                }

            q = questions[key]
            if obj.get("is_answerable") != q["first_is_answerable"]:
                q["is_answerable_mismatch"] += 1
            rs = obj.get("review_snippets", [])
            if isinstance(rs, list) and rs != q["first_review_snippets"]:
                q["review_snippets_mismatch_count"] += 1

            answers = obj.get("answers", [])
            if isinstance(answers, list):
                q["answers"].extend(answers)
            q["uids"].append(uid)
            q["n_dong_gop"] += 1

    question_list = []
    for (asin, qnorm), q in questions.items():
        q["is_answerable_auto"] = q["first_is_answerable"]
        q["review_snippets_cau_hoi"] = q["first_review_snippets"]
        if q["answers"]:
            q["dap_an_khong_biet"] = all(
                is_dont_know(a.get("answerText", ""))
                for a in q["answers"]
                if isinstance(a, dict)
            )
        else:
            q["dap_an_khong_biet"] = False
        q["doc_id"] = f"doc-{asin}"
        question_list.append(q)

    questions_dev = [q for q in question_list if q["asin"] in asin_dev_set]
    questions_test = [q for q in question_list if q["asin"] in asin_test_set]
    questions_dev.sort(key=lambda q: q["asin"])
    questions_test.sort(key=lambda q: q["asin"])

    docs_dev = [docs[a] for a in asin_dev if a in docs]
    docs_test = [docs[a] for a in asin_test if a in docs]

    docs_dev_out = []
    for d in docs_dev:
        docs_dev_out.append(
            {
                "doc_id": d["doc_id"],
                "asin": d["asin"],
                "category": d["category"],
                "nguon_goc": d["nguon_goc"],
                "snippets": d["snippets"],
                "text_goc": d["text_goc"],
                "n_snippets": d["n_snippets"],
                "n_chars": d["n_chars"],
                "n_questions": d["n_questions"],
            }
        )
    docs_test_out = []
    for d in docs_test:
        docs_test_out.append(
            {
                "doc_id": d["doc_id"],
                "asin": d["asin"],
                "category": d["category"],
                "nguon_goc": d["nguon_goc"],
                "snippets": d["snippets"],
                "text_goc": d["text_goc"],
                "n_snippets": d["n_snippets"],
                "n_chars": d["n_chars"],
                "n_questions": d["n_questions"],
            }
        )

    for i, q in enumerate(questions_dev, start=1):
        q["qid_moi"] = f"q-dev-{i:04d}"
    for i, q in enumerate(questions_test, start=1):
        q["qid_moi"] = f"q-test-{i:04d}"

    questions_dev_out = []
    for q in questions_dev:
        questions_dev_out.append(
            {
                "qid_moi": q["qid_moi"],
                "asin": q["asin"],
                "doc_id": q["doc_id"],
                "question_text": q["question_text"],
                "question_type": q["question_type"],
                "is_answerable_auto": q["is_answerable_auto"],
                "answers": q["answers"],
                "uids": q["uids"],
                "review_snippets_cau_hoi": q["review_snippets_cau_hoi"],
                "dap_an_khong_biet": q["dap_an_khong_biet"],
            }
        )
    questions_test_out = []
    for q in questions_test:
        questions_test_out.append(
            {
                "qid_moi": q["qid_moi"],
                "asin": q["asin"],
                "doc_id": q["doc_id"],
                "question_text": q["question_text"],
                "question_type": q["question_type"],
                "is_answerable_auto": q["is_answerable_auto"],
                "answers": q["answers"],
                "uids": q["uids"],
                "review_snippets_cau_hoi": q["review_snippets_cau_hoi"],
                "dap_an_khong_biet": q["dap_an_khong_biet"],
            }
        )

    write_jsonl(processed_dir / "docs_dev.jsonl", docs_dev_out)
    write_jsonl(processed_dir / "docs_test.jsonl", docs_test_out)
    write_jsonl(processed_dir / "questions_dev.jsonl", questions_dev_out)
    write_jsonl(processed_dir / "questions_test.jsonl", questions_test_out)

    csv_path = report_dir / "ung_vien_kiem_tay.csv"
    nhom = defaultdict(list)
    for q in questions_dev:
        nhom[(q["question_type"], str(q["is_answerable_auto"]))].append(q)

    cac_nhom = sorted(nhom.keys())
    so_nhom = len(cac_nhom)
    quota = n_kiem_tay // so_nhom if so_nhom else 0
    du = n_kiem_tay % so_nhom if so_nhom else 0

    kiem_tay_rows = []
    for i, nhom_key in enumerate(cac_nhom):
        ds = list(nhom[nhom_key])
        rng.shuffle(ds)
        lay = quota + (1 if i < du else 0)
        lay = min(lay, len(ds))
        for q in ds[:lay]:
            answers_str = " || ".join(
                (
                    a.get("answerText", "")[:300]
                    if isinstance(a, dict)
                    else str(a)[:300]
                )
                for a in q["answers"]
            )
            snippets_str = " ||| ".join(q["review_snippets_cau_hoi"])
            kiem_tay_rows.append(
                {
                    "qid_moi": q["qid_moi"],
                    "asin": q["asin"],
                    "question_text": q["question_text"],
                    "question_type": q["question_type"],
                    "is_answerable_auto": q["is_answerable_auto"],
                    "answers_noi_bang": answers_str,
                    "snippets_cua_cau_hoi": snippets_str,
                    "nhan_tay": "",
                    "ghi_chu": "",
                }
            )

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "qid_moi",
            "asin",
            "question_text",
            "question_type",
            "is_answerable_auto",
            "answers_noi_bang",
            "snippets_cua_cau_hoi",
            "nhan_tay",
            "ghi_chu",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kiem_tay_rows)

    chars_dev = stats_of([d["n_chars"] for d in docs_dev_out])
    snip_dev = stats_of([d["n_snippets"] for d in docs_dev_out])
    chars_test = stats_of([d["n_chars"] for d in docs_test_out])
    snip_test = stats_of([d["n_snippets"] for d in docs_test_out])

    qt_dev = defaultdict(int)
    qt_test = defaultdict(int)
    ia_dev = defaultdict(int)
    ia_test = defaultdict(int)
    dap_an_kb_dev = 0
    dap_an_kb_test = 0
    for q in questions_dev_out:
        qt_dev[q["question_type"]] += 1
        ia_dev[str(q["is_answerable_auto"])] += 1
        if q["dap_an_khong_biet"]:
            dap_an_kb_dev += 1
    for q in questions_test_out:
        qt_test[q["question_type"]] += 1
        ia_test[str(q["is_answerable_auto"])] += 1
        if q["dap_an_khong_biet"]:
            dap_an_kb_test += 1

    asin_dev_out_set = set(d["asin"] for d in docs_dev_out)
    asin_test_out_set = set(d["asin"] for d in docs_test_out)
    giao = asin_dev_out_set & asin_test_out_set
    kiem_tra_a = len(giao) == 0

    doc_ids = set(d["doc_id"] for d in docs_dev_out) | set(
        d["doc_id"] for d in docs_test_out
    )
    kiem_tra_b = all(
        q["doc_id"] in doc_ids for q in questions_dev_out + questions_test_out
    )

    kiem_tra_c = len(asin_test_out_set & asin_chung) == 0

    asin_source_check = {}
    kiem_tra_d = True
    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            asin = obj.get("asin", "")
            source = obj.get("nguon_goc")
            if asin in asin_dev_set or asin in asin_test_set:
                if asin in asin_source_check and asin_source_check[asin] != source:
                    kiem_tra_d = False
                asin_source_check[asin] = source

    lines = []
    lines.append("=" * 70)
    lines.append("BÁO CÁO ĐỦ LÀM VIỆC NHỎ TỪ AMAZONQA V2")
    lines.append("=" * 70)
    lines.append(f"File nguồn: {merged_path}")
    lines.append(f"Danh mục: {category}")
    lines.append(f"Seed: {seed}")
    lines.append("")
    lines.append("Bước 1: asin chung val & test")
    lines.append(f"  Số asin chung: {n_asin_chung}")
    lines.append(f"  Số dòng test bị loại: {n_test_lines_loai}")
    if n_asin_chung != 15 or n_test_lines_loai != 115:
        lines.append("  CẢNH BÁO: khác kỳ vọng (15 asin, 115 dòng)")
    lines.append("")
    lines.append("Bước 4: chọn mẫu tài liệu")
    lines.append(
        f"  Asin dev đủ điều kiện (n_chars >= {min_doc_chars}): {len(docs_dev_candidates)}"
    )
    lines.append(
        f"  Asin test đủ điều kiện: {len(docs_test_candidates)}"
    )
    lines.append(f"  Asin dev được chọn: {len(asin_dev)}")
    lines.append(f"  Asin test được chọn: {len(asin_test)}")
    lines.append("")
    lines.append("Bước 5: câu hỏi")
    lines.append(f"  Số câu hỏi dev sau gộp trùng: {len(questions_dev_out)}")
    lines.append(f"  Số câu hỏi test sau gộp trùng: {len(questions_test_out)}")
    lines.append(
        f"  Số câu bị loại do questionText < {min_question_chars} ký tự: {n_short_question}"
    )
    lines.append("")
    lines.append("Phân bố question_type (dev):")
    for k, v in sorted(qt_dev.items()):
        lines.append(f"  {k}: {v}")
    lines.append("Phân bố question_type (test):")
    for k, v in sorted(qt_test.items()):
        lines.append(f"  {k}: {v}")
    lines.append("Phân bố is_answerable_auto (dev):")
    for k, v in sorted(ia_dev.items()):
        lines.append(f"  {k}: {v}")
    lines.append("Phân bố is_answerable_auto (test):")
    for k, v in sorted(ia_test.items()):
        lines.append(f"  {k}: {v}")
    lines.append(f"Số câu gắn cờ dap_an_khong_biet (dev): {dap_an_kb_dev}")
    lines.append(f"Số câu gắn cờ dap_an_khong_biet (test): {dap_an_kb_test}")
    lines.append("")
    lines.append("Thống kê tài liệu dev:")
    lines.append(f"  Số tài liệu: {len(docs_dev_out)}")
    lines.append(
        f"  n_chars: min={chars_dev[0]}, trung vị={chars_dev[1]:.1f}, "
        f"p90={chars_dev[2]:.1f}, max={chars_dev[3]}"
    )
    lines.append(
        f"  n_snippets: min={snip_dev[0]}, trung vị={snip_dev[1]:.1f}, "
        f"p90={snip_dev[2]:.1f}, max={snip_dev[3]}"
    )
    lines.append("Thống kê tài liệu test:")
    lines.append(f"  Số tài liệu: {len(docs_test_out)}")
    lines.append(
        f"  n_chars: min={chars_test[0]}, trung vị={chars_test[1]:.1f}, "
        f"p90={chars_test[2]:.1f}, max={chars_test[3]}"
    )
    lines.append(
        f"  n_snippets: min={snip_test[0]}, trung vị={snip_test[1]:.1f}, "
        f"p90={snip_test[2]:.1f}, max={snip_test[3]}"
    )
    lines.append("")
    lines.append("Kiểm tra tự động:")
    lines.append(
        f"  (a) asin dev giao asin test = rỗng: "
        f"{'ĐẠT' if kiem_tra_a else 'KHÔNG ĐẠT'} (giao: {len(giao)})"
    )
    lines.append(
        f"  (b) mọi câu hỏi đều có doc_id tồn tại: "
        f"{'ĐẠT' if kiem_tra_b else 'KHÔNG ĐẠT'}"
    )
    lines.append(
        f"  (c) không asin test nằm trong danh sách asin chung đã loại: "
        f"{'ĐẠT' if kiem_tra_c else 'KHÔNG ĐẠT'}"
    )
    lines.append(
        f"  (d) mọi dòng nguồn của asin được chọn đều thuộc đúng một trong hai tập: "
        f"{'ĐẠT' if kiem_tra_d else 'KHÔNG ĐẠT'}"
    )
    lines.append("")
    lines.append("=" * 70)
    lines.append("KẾT THÚC BÁO CÁO")
    lines.append("=" * 70)

    text = "\n".join(lines) + "\n"
    print(text, end="")
    report_path = report_dir / "bo_lam_viec.md"
    report_path.write_text(text, encoding="utf-8")
    logger.info(f"Đã ghi báo cáo vào {report_path}")
    logger.info(f"Đã ghi file kiểm tay vào {csv_path}")

    return {
        "n_asin_chung": n_asin_chung,
        "n_test_lines_loai": n_test_lines_loai,
        "n_docs_dev": len(docs_dev_out),
        "n_docs_test": len(docs_test_out),
        "n_questions_dev": len(questions_dev_out),
        "n_questions_test": len(questions_test_out),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }
