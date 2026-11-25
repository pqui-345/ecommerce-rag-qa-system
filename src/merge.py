"""Mô-đun gộp hai tập dữ liệu val và test theo phương pháp đọc luồng (streaming).

Phiên bản sửa lỗi: khử trùng lặp theo hash nội dung (sha1) thay vì theo qid,
vì qid chỉ là số thứ tự trong từng file (val: 0..N, test: 0..M) nên trùng qid
không có nghĩa là trùng dòng. Mỗi dòng ghi ra thêm trường "nguon_goc" và
"uid" = f"{nguon_goc}-{qid}".
"""

import hashlib
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class SafeStreamHandler(logging.StreamHandler):
    """Handler an toàn chống lỗi UnicodeEncodeError trên console Windows."""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            try:
                encoding = getattr(stream, "encoding", "utf-8") or "utf-8"
                encoded = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(encoded + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


def setup_logger(log_file: Path) -> logging.Logger:
    """Khởi tạo logger ghi đồng thời ra file log UTF-8 và màn hình console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gop_du_lieu")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = SafeStreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def check_disk_space(val_path: Path, test_path: Path, out_path: Path, logger: logging.Logger) -> None:
    """Kiểm tra dung lượng đĩa trống trước khi ghi file đầu ra."""
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(out_dir).free
    estimated_needed = 0
    for p in [val_path, test_path]:
        if p.exists():
            estimated_needed += p.stat().st_size
    estimated_needed = int(estimated_needed * 1.1)

    logger.info(f"Dung lượng đĩa trống tại '{out_dir}': {free_bytes / (1024**3):.2f} GB")
    logger.info(f"Dung lượng dự kiến cần tối thiểu: {estimated_needed / (1024**3):.2f} GB")

    if free_bytes < estimated_needed:
        msg = (
            f"DỪNG: Không đủ dung lượng đĩa trống! "
            f"Còn lại: {free_bytes / (1024**3):.2f} GB, cần tối thiểu: {estimated_needed / (1024**3):.2f} GB."
        )
        logger.error(msg)
        raise RuntimeError(msg)


def content_hash(record: dict) -> str:
    """Mã hoá sha1 của đối tượng gốc (không có nguon_goc và uid)."""
    return hashlib.sha1(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def norm_qtext(value) -> str:
    """Chuẩn hóa questionText: strip + lowercase."""
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def gop_du_lieu(
    val_path: Path,
    test_path: Path,
    out_path: Path,
    dedup_key: str = "qid",
    log_path: Path = None,
    report_path: Path = None,
) -> dict:
    """
    Gộp lần lượt file val rồi test theo luồng (streaming), giữ nguyên các trường gốc,
    bổ sung 'nguon_goc' (và 'uid' khi dùng hash nội dung), loại trùng và xử lý an toàn
    dòng JSON hỏng.

    Nếu dedup_key == "hash_noi_dung": loại trùng bằng sha1 của toàn bộ bản ghi gốc
    (trừ nguon_goc/uid), chỉ lưu hash trong RAM, giữ dòng đầu tiên gặp. Mỗi dòng thêm
    uid = f"{nguon_goc}-{qid}"; uid phải duy nhất, ngược lại dừng và báo lỗi.
    Ngược lại: giữ nguyên hành vi cũ (loại trùng theo khóa dedup_key, mặc định qid).
    """
    if log_path is None:
        log_path = Path("logs/gop_du_lieu.log")

    logger = setup_logger(log_path)
    use_hash = dedup_key == "hash_noi_dung"
    method_label = "hash noi dung (sha1)" if use_hash else dedup_key

    logger.info("Bắt đầu tiến trình gộp dữ liệu (phiên bản sửa lỗi)...")
    logger.info(f"- File val: {val_path}")
    logger.info(f"- File test: {test_path}")
    logger.info(f"- File đầu ra: {out_path}")
    logger.info(f"- Phương pháp loại trùng: {method_label}")

    check_disk_space(val_path, test_path, out_path, logger)

    seen_hashes = set() if use_hash else None
    seen_keys = set() if not use_hash else None
    seen_uids = set()

    stats = {
        "val": {"read": 0, "corrupted": 0, "written": 0},
        "test": {"read": 0, "corrupted": 0, "written": 0},
        "dedup_dropped": 0,
        "missing_key": 0,
        "missing_qid": 0,
        "uid_error": None,
        "total_written": 0,
        "out_size_bytes": 0,
        "val_asins": set(),
        "test_asins": set(),
        "val_pairs": set(),
        "test_pairs": set(),
        "val_pair_counter": Counter(),
        "test_pair_counter": Counter(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    uid_unique = True

    with open(out_path, "w", encoding="utf-8") as out_f:
        for source_name, file_path in [("val", val_path), ("test", test_path)]:
            if not file_path.exists():
                err = f"Không tìm thấy file nguồn: {file_path}"
                logger.error(err)
                raise FileNotFoundError(err)

            logger.info(f"Đang đọc dữ liệu luồng từ '{file_path.name}' ({source_name})...")
            with open(file_path, "r", encoding="utf-8", errors="replace") as in_f:
                for line_no, line in enumerate(in_f, start=1):
                    stats[source_name]["read"] += 1
                    line_str = line.strip()
                    if not line_str:
                        continue

                    try:
                        record = json.loads(line_str)
                    except Exception as e:
                        stats[source_name]["corrupted"] += 1
                        logger.warning(
                            f"Dòng JSON hỏng tại file '{file_path.name}', dòng {line_no}: {e}"
                        )
                        continue

                    if not isinstance(record, dict):
                        stats[source_name]["corrupted"] += 1
                        continue

                    asin = record.get("asin", "")
                    pair = (asin, norm_qtext(record.get("questionText", "")))
                    if source_name == "val":
                        stats["val_asins"].add(asin)
                        stats["val_pairs"].add(pair)
                        stats["val_pair_counter"][pair] += 1
                    else:
                        stats["test_asins"].add(asin)
                        stats["test_pairs"].add(pair)
                        stats["test_pair_counter"][pair] += 1

                    if use_hash:
                        qid = record.get("qid")
                        if qid is None:
                            stats["missing_qid"] += 1
                            err = (
                                f"Thiếu qid tại file '{file_path.name}', dòng {line_no}; "
                                f"không thể tạo uid."
                            )
                            logger.error(err)
                            stats["uid_error"] = err
                            uid_unique = False
                            raise RuntimeError(err)
                        uid = f"{source_name}-{qid}"
                        if uid in seen_uids:
                            err = f"UID trùng '{uid}' tại file '{file_path.name}', dòng {line_no}. Dừng lại."
                            logger.error(err)
                            stats["uid_error"] = err
                            uid_unique = False
                            raise RuntimeError(err)
                        seen_uids.add(uid)

                    if use_hash:
                        h = content_hash(record)
                        if h in seen_hashes:
                            stats["dedup_dropped"] += 1
                            continue
                        seen_hashes.add(h)
                    else:
                        if dedup_key in record:
                            key_val = record[dedup_key]
                            if key_val in seen_keys:
                                stats["dedup_dropped"] += 1
                                continue
                            seen_keys.add(key_val)
                        else:
                            stats["missing_key"] += 1

                    record["nguon_goc"] = source_name
                    if use_hash:
                        record["uid"] = uid

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats[source_name]["written"] += 1
                    stats["total_written"] += 1

    stats["out_size_bytes"] = out_path.stat().st_size

    val_written = stats["val"]["written"]
    test_written = stats["test"]["written"]
    total_read = stats["val"]["read"] + stats["test"]["read"]
    total_corrupted = stats["val"]["corrupted"] + stats["test"]["corrupted"]
    expected = total_read - total_corrupted

    val_asins = stats["val_asins"]
    test_asins = stats["test_asins"]
    asin_common = len(val_asins & test_asins)

    val_pairs = stats["val_pairs"]
    test_pairs = stats["test_pairs"]
    pair_common = len(val_pairs & test_pairs)

    val_pair_dup = sum(c - 1 for c in stats["val_pair_counter"].values() if c > 1)
    test_pair_dup = sum(c - 1 for c in stats["test_pair_counter"].values() if c > 1)

    uid_status = "KHỚP" if uid_unique else "LỆCH"

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BAO CAO GOP DU LIEU V2 (AMAZONQA)")
    report_lines.append("=" * 70)
    report_lines.append(f"- Phuong phap loai trung: {method_label}")
    report_lines.append(f"- File val: {val_path}")
    report_lines.append(f"  So dong doc: {stats['val']['read']:,} (hong: {stats['val']['corrupted']:,}, ghi: {val_written:,})")
    report_lines.append(f"- File test: {test_path}")
    report_lines.append(f"  So dong doc: {stats['test']['read']:,} (hong: {stats['test']['corrupted']:,}, ghi: {test_written:,})")
    report_lines.append(f"- Tong dong JSON hong: val={stats['val']['corrupted']:,}, test={stats['test']['corrupted']:,}")
    report_lines.append(f"- So dong trung chinh xac (hash noi dung) bo loc: {stats['dedup_dropped']:,}")
    report_lines.append(f"- Tong so da ghi: {stats['total_written']:,} (ky vong ≈ {expected:,})")
    if abs(stats["total_written"] - expected) > max(100, expected * 0.01):
        report_lines.append(
            f"  CANH BAO: so da ghi lech ky vong {abs(stats['total_written'] - expected):,} dong."
        )
    report_lines.append(f"- Dung luong file dau ra: {stats['out_size_bytes']:,} bytes ({stats['out_size_bytes'] / (1024 * 1024):.2f} MiB)")
    report_lines.append("- Dem theo nguon_goc:")
    report_lines.append(f"    val: {val_written:,}")
    report_lines.append(f"    test: {test_written:,}")
    report_lines.append("- So asin khac biet:")
    report_lines.append(f"    val: {len(val_asins):,}, test: {len(test_asins):,}, chung: {asin_common:,}")
    report_lines.append(f"- So cap (asin, questionText_chuan) xuat hop hai file: {pair_common:,}")
    report_lines.append("- So cap (asin, questionText) lap noi bo moi file:")
    report_lines.append(f"    val: {val_pair_dup:,}, test: {test_pair_dup:,}")
    report_lines.append(f"- Kiem tra uid duy nhat: {uid_status}")
    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines) + "\n"

    logger.info("\n" + report_text)

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")

    return stats
