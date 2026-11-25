"""
Kiem tra du lieu AmazonQA da gop (val, test, merged).
- Chi su dung thu vien chuan: json, hashlib, collections, pathlib, argparse (va sys).
- Doc tung dong (streaming). Khong giu noi dung dong trong RAM: voi moi qid chi luu hash
  (sha1 cua json.dumps(obj, sort_keys=True, ensure_ascii=False)).
- Dong JSON hong: dem va bo qua, khong lam sap chuong trinh.
- Dat stdout ve UTF-8 (chay tren Windows).
- Khong co side-effect khi import; co ham main() va if __name__ == "__main__".
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_VAL = ROOT / "data" / "raw" / "val-qar.jsonl"
DEFAULT_TEST = ROOT / "data" / "raw" / "test-qar.jsonl"
DEFAULT_MERGED = ROOT / "data" / "merged" / "amazonqa_merged.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "kiem_tra_qid.md"


def obj_hash(obj):
    """Ma hoa sha1 cua toan bo doi tuong duoc serialize (giu nguyen thu tu khoa, khong thoat unicode)."""
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def get_qid(obj):
    """Lay gia tri qid. Tra ve None neu thieu hoac rong."""
    if not isinstance(obj, dict):
        return None
    qid = obj.get("qid")
    if qid is None:
        return None
    if isinstance(qid, str) and qid == "":
        return None
    return qid


def scan_file(path):
    """
    Doc mot file JSONL luot thong qua streaming.
    Tra ve dict thong ke. Khong giu noi dung dong trong RAM (ngoai exception first_obj,
    mot dong duyets, duoc luu tam de phan truc cau truc o muc 6).
    """
    stats = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_mib": path.stat().st_size / (1024.0 * 1024.0),
        "lines": 0,
        "bad_lines": 0,
        "qid_hash": {},
        "qid_seen": set(),
        "dup_qids": set(),
        "dup_lines": 0,
        "missing_qid": 0,
        "first_obj": None,
        "source_counter": Counter(),
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stats["lines"] += 1
            try:
                obj = json.loads(line)
            except Exception:
                stats["bad_lines"] += 1
                continue
            if stats["first_obj"] is None:
                stats["first_obj"] = obj
            qid = get_qid(obj)
            if qid is None:
                stats["missing_qid"] += 1
            else:
                if qid in stats["qid_seen"]:
                    stats["dup_qids"].add(qid)
                    stats["dup_lines"] += 1
                else:
                    stats["qid_seen"].add(qid)
                    stats["qid_hash"][qid] = obj_hash(obj)
            if isinstance(obj, dict):
                stats["source_counter"][obj.get("nguon_goc", "KHONG_CO")] += 1
            else:
                stats["source_counter"]["KHONG_CO"] += 1
    return stats


def find_objects(path, target_qids):
    """
    Lay lai doi tuong toan bo cho cac qid trong target_qids (toi da 5) bang mot luot doc
    stream. Dung chi cho viec so sanh chi tiet cac cap khac nhau o muc 4.
    """
    target = set(target_qids)
    found = {}
    if not target:
        return found
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = get_qid(obj)
            if qid is not None and qid in target and qid not in found:
                found[qid] = obj
                if len(found) == len(target):
                    break
    return found


def diff_keys(o1, o2):
    """Tra ve danh sach ten khoa co gia tri khac nhau giua hai doi tuong."""
    all_keys = set(o1) | set(o2)
    res = []
    for k in sorted(all_keys, key=str):
        if k not in o1:
            res.append((k, "thieu_trong_file_1"))
        elif k not in o2:
            res.append((k, "thieu_trong_file_2"))
        elif o1[k] != o2[k]:
            res.append((k, "khac_gia_tri"))
    return res


def truncate(val, max_len=150):
    """Cat ngat chuoi dai va danh sach de in, khong in nguyen van chuoi dai."""
    if isinstance(val, str):
        return val if len(val) <= max_len else val[:max_len] + "..."
    if isinstance(val, list):
        head = ", ".join(truncate(x, max_len) for x in val[:2])
        return "[%s] ... (%d phan tu)" % (head, len(val))
    if isinstance(val, dict):
        items = ", ".join(
            "%s: %s" % (k, truncate(v, max_len)) for k, v in list(val.items())[:3]
        )
        return "{%s} ... (%d khoa)" % (items, len(val))
    return str(val)


def analyze_merged(path, first_keys):
    """
    Mot luot doc stream tren file gop de tinh:
    - so khoa bi thieu moi dong (so voi tap khoa cua dong dau)
    - phan bo category (top 20, so cau hoi va so asin khac biet)
    - ty le is_answerable va questionType
    """
    missing_counter = Counter()
    cat_counter = Counter()
    cat_asins = defaultdict(set)
    ans_counter = Counter()
    qt_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for k in first_keys:
                if k not in obj:
                    missing_counter[k] += 1
            cat = obj.get("category")
            if cat is not None:
                cat_counter[cat] += 1
                asin = obj.get("asin")
                if asin is not None:
                    cat_asins[cat].add(asin)
            ia = obj.get("is_answerable")
            if ia is True or ia == 1:
                ans_counter["True"] += 1
            elif ia is False or ia == 0:
                ans_counter["False"] += 1
            else:
                ans_counter["missing"] += 1
            qt = obj.get("questionType")
            qt_counter[str(qt) if qt is not None else "missing"] += 1
    return missing_counter, cat_counter, cat_asins, ans_counter, qt_counter


def build_output(val_path, test_path, merged_path, report_path):
    out = []

    def emit(s=""):
        out.append(s)

    emit("=" * 70)
    emit("MUC 1: DUNG LUONG VA SO DONG CUA BA FILE")
    emit("=" * 70)
    vs = scan_file(val_path)
    ts = scan_file(test_path)
    ms = scan_file(merged_path)

    for label, st in [("val", vs), ("test", ts), ("merged", ms)]:
        emit("- File %s: %s" % (label, st["path"]))
        emit("  Dung luong: %.2f MiB (%d bytes)" % (st["size_mib"], st["size_bytes"]))
        emit("  So dong: %d" % st["lines"])
        emit("  So dong JSON hong: %d" % st["bad_lines"])
        emit("")

    emit("=" * 70)
    emit("MUC 2: VAL VA TEST - QID, LAP NOI BO, THIEU QID")
    emit("=" * 70)
    for label, st in [("val", vs), ("test", ts)]:
        qids = set(st["qid_hash"].keys())
        emit("- File %s:" % label)
        emit("  So qid khac biet: %d" % len(qids))
        emit("  So qid xuat hien nhieu hon 1 lan (lap noi bo): %d" % len(st["dup_qids"]))
        emit("  (Tong so dong bi lap qua: %d)" % st["dup_lines"])
        emit("  So dong thieu qid hoac qid rong: %d" % st["missing_qid"])
        emit("")

    val_qids = set(vs["qid_hash"].keys())
    test_qids = set(ts["qid_hash"].keys())

    emit("=" * 70)
    emit("MUC 3: QUAN HE VAL VA TEST")
    emit("=" * 70)
    common = val_qids & test_qids
    val_only = val_qids - test_qids
    test_only = test_qids - val_qids
    emit("- So qid chung: %d" % len(common))
    emit("- So qid chi co tai val: %d" % len(val_only))
    emit("- So qid chi co tai test: %d" % len(test_only))
    emit("")

    emit("=" * 70)
    emit("MUC 4: CAC QID CHUNG - SO SANH HASH")
    emit("=" * 70)
    same_count = 0
    diff_count = 0
    diff_qids = []
    for qid in common:
        if vs["qid_hash"].get(qid) == ts["qid_hash"].get(qid):
            same_count += 1
        else:
            diff_count += 1
            diff_qids.append(qid)
    emit("- Tong so qid chung: %d" % len(common))
    emit("- So cap hash giong nhau: %d" % same_count)
    emit("- So cap hash KHAC NHAU: %d" % diff_count)
    if diff_count > 0:
        emit("- Toi da 5 vi du cap khac nhau (qid va ten khoa khac):")
        val_objs = find_objects(val_path, diff_qids[:5])
        test_objs = find_objects(test_path, diff_qids[:5])
        shown = 0
        for qid in diff_qids:
            if shown >= 5:
                break
            o1 = val_objs.get(qid)
            o2 = test_objs.get(qid)
            if o1 is None or o2 is None:
                continue
            dks = diff_keys(o1, o2)
            emit("  * qid = %s: %d khoa khac nhau -> %s" % (qid, len(dks), ", ".join(k for k, _ in dks)))
            shown += 1
    emit("")

    emit("=" * 70)
    emit("MUC 5: FILE GOP - QID, NGUON_GOC, KIEM TRA HOP")
    emit("=" * 70)
    merged_qids = set(ms["qid_hash"].keys())
    union_qids = val_qids | test_qids
    emit("- So dong: %d" % ms["lines"])
    emit("- So qid khac biet: %d" % len(merged_qids))
    emit("- So qid lap: %d (distinct qid lap: %d, tong dong lap qua: %d)" % (
        len(ms["dup_qids"]), len(ms["dup_qids"]), ms["dup_lines"]))
    emit("- Dem theo nguon_goc:")
    for src, cnt in ms["source_counter"].most_common():
        emit("    %s: %d" % (src, cnt))
    extra = merged_qids - union_qids
    missing = union_qids - merged_qids
    if not extra and not missing:
        hop = "KHOP"
        detail = "tap qid file gop bang hop (val U test)"
    else:
        hop = "LECH"
        detail = "thua: %d, thieu: %d" % (len(extra), len(missing))
    emit("- Kiem tra tap qid file gop == val U test: %s" % hop)
    emit("  %s" % detail)
    emit("")

    emit("=" * 70)
    emit("MUC 6: CAU TRUC MOT DONG (FILE GOP)")
    emit("=" * 70)
    first_obj = ms["first_obj"]
    if first_obj is None:
        emit("Khong co dong hop le nao trong file gop.")
    else:
        first_keys = set(first_obj.keys())
        emit("- Dong dau tien: %s" % ms["path"])
        emit("  Danh sach khoa va kieu du lieu:")
        for k in sorted(first_keys, key=str):
            emit("    %s (%s)" % (k, type(first_obj[k]).__name__))
        emit("  Noi dung dong dau (chuoi cat toi da 150 ky tu; list in 2 phan tu dau + so luong):")
        for k in sorted(first_keys, key=str):
            emit("    %s = %s" % (k, truncate(first_obj[k], 150)))

        missing_counter, _, _, _, _ = analyze_merged(merged_path, first_keys)
        emit("  Dem kho thieu moi dong (so voi tap khoa cua dong dau):")
        for k in sorted(first_keys, key=str):
            emit("    %s: thieu o %d dong" % (k, missing_counter.get(k, 0)))
    emit("")

    emit("=" * 70)
    emit("MUC 7: THONG KE NHÊ (FILE GOP)")
    emit("=" * 70)
    missing_counter, cat_counter, cat_asins, ans_counter, qt_counter = analyze_merged(
        merged_path, set(first_obj.keys()) if first_obj is not None else set()
    )

    emit("- Phan bo category (top 20, so cau hoi / so asin khac biet):")
    total_qa = sum(cat_counter.values())
    for cat, cnt in cat_counter.most_common(20):
        emit("    %-40s %d cau hoi / %d asin khac biet" % (cat, cnt, len(cat_asins.get(cat, set()))))
    emit("  Tong cac category da dem: %d (tong so cau hoi: %d)" % (len(cat_counter), total_qa))

    emit("- Ty le is_answerable:")
    ans_total = sum(ans_counter.values())
    for k in ["True", "False", "missing"]:
        emit("    %s: %d (%.2f%%)" % (k, ans_counter.get(k, 0), 100.0 * ans_counter.get(k, 0) / ans_total if ans_total else 0.0))

    emit("- Ty le questionType:")
    qt_total = sum(qt_counter.values())
    for k, cnt in qt_counter.most_common():
        emit("    %s: %d (%.2f%%)" % (k, cnt, 100.0 * cnt / qt_total if qt_total else 0.0))
    emit("")

    emit("=" * 70)
    emit("KET LUAN")
    emit("=" * 70)
    emit("(a) Val co nam tron trong test khong: %s (qid chi co tai val = %d)" % (
        "CO" if not val_only else "KHONG", len(val_only)))
    emit("(b) File gop co khop hop qid khong: %s (%s)" % (hop, detail))
    emit("(c) Co phat hien bat thuong nao khac khong: %s" % (
        "CO - %d cap qid chung co hash khac nhau; file gop chua %d qid nguon_goc=val va %d qid nguon_goc=test" % (
            diff_count, ms["source_counter"].get("val", 0), ms["source_counter"].get("test", 0)
        ) if (diff_count or ms["source_counter"].get("val", 0) or ms["source_counter"].get("test", 0))
        else "KHONG - tat ca chu trinh khong co hieu kiem tra"
    ))
    emit("")
    emit("=" * 70)
    emit("KET THUC BAO CAO")
    emit("=" * 70)

    text = "\n".join(out) + "\n"

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")

    return text


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Kiem tra du lieu AmazonQA da gop (val, test, merged)."
    )
    parser.add_argument("--val", default=str(DEFAULT_VAL), help="Duong dan file val-qar.jsonl")
    parser.add_argument("--test", default=str(DEFAULT_TEST), help="Duong dan file test-qar.jsonl")
    parser.add_argument("--merged", default=str(DEFAULT_MERGED), help="Duong dan file amazonqa_merged.jsonl")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Duong dan file bao ra ket qua")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")

    val_path = Path(args.val)
    test_path = Path(args.test)
    merged_path = Path(args.merged)
    report_path = Path(args.report)

    text = build_output(val_path, test_path, merged_path, report_path)
    print(text, end="")


if __name__ == "__main__":
    main()
