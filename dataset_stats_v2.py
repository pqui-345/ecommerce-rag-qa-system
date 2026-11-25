"""
Thong ke tren data/merged/amazonqa_merged_v2.jsonl (doc streaming, KHONG sua file).
Chi dung thu vien chuan. Ghi ket quả vao reports/thong_ke_v2.md va in ra man hinh.
"""

import hashlib
import json
import statistics
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MERGED_V2 = ROOT / "data" / "merged" / "amazonqa_merged_v2.jsonl"
REPORT = ROOT / "reports" / "thong_ke_v2.md"

sys.stdout.reconfigure(encoding="utf-8")

OUT = []


def emit(s=""):
    OUT.append(s)
    print(s)


def pct(n, total):
    return (100.0 * n / total) if total else 0.0


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    k = (q / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def norm_qtext(t):
    if not isinstance(t, str):
        t = str(t)
    return t.strip().lower()


def norm_segment(s):
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.strip().lower().split())


def seg_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def truncate(v, n=120):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "..."
    if isinstance(v, list):
        head = ", ".join(truncate(x, n) for x in v[:2])
        return f"[{head}] ... ({len(v)} phan tu)"
    if isinstance(v, dict):
        items = ", ".join(
            f"{k}: {truncate(v2, n)}" for k, v2 in list(v.items())[:3]
        )
        return f"{{{items}}} ... ({len(v)} khoa)"
    return repr(v)


def keys_diff(objs):
    """Tra ve danh sach ten khóa co gia tri khong dong nhat giua cac doi tuong."""
    all_keys = set()
    for o in objs:
        all_keys.update(o.keys())
    diff = []
    for k in sorted(all_keys, key=str):
        vals = [o.get(k, "<MISSING>") for o in objs]
        first = vals[0]
        if any(v != first for v in vals[1:]):
            diff.append(k)
    return diff


emit("=" * 70)
emit("PASS 1: Doc stream thong ke co ban (asin, pair, category, is_answerable, do dai)")
emit("=" * 70)

asin_source_count = defaultdict(lambda: defaultdict(int))
pair_source_count = defaultdict(lambda: defaultdict(int))
category_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "asins": set()}))
category_asin_count = defaultdict(
    lambda: defaultdict(lambda: defaultdict(int))
)
is_ans_cat = defaultdict(lambda: defaultdict(Counter))
is_ans_qt = defaultdict(lambda: defaultdict(Counter))
count_answer_ge1 = defaultdict(int)
length_qtext = []
length_answer = []
asin_category = {}

total_lines = 0
bad_lines = 0

with open(MERGED_V2, "r", encoding="utf-8") as f:
    for line in f:
        total_lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            bad_lines += 1
            continue
        if not isinstance(obj, dict):
            bad_lines += 1
            continue

        source = obj.get("nguon_goc", "?")
        asin = obj.get("asin", "")
        category = obj.get("category", "")
        qtext = obj.get("questionText", "")
        qnorm = norm_qtext(qtext)
        qid = obj.get("qid")

        asin_source_count[asin][source] += 1
        pair_source_count[(asin, qnorm)][source] += 1

        cs = category_stats[source][category]
        cs["count"] += 1
        cs["asins"].add(asin)
        category_asin_count[source][category][asin] += 1
        asin_category[asin] = category

        ia = obj.get("is_answerable")
        is_ans_cat[source][category][str(ia)] += 1
        is_ans_qt[source][str(obj.get("questionType", ""))][str(ia)] += 1

        ans = obj.get("answers", [])
        if isinstance(ans, list) and len(ans) >= 1:
            count_answer_ge1[source] += 1

        length_qtext.append(len(qtext) if isinstance(qtext, str) else 0)
        if isinstance(ans, list):
            for a in ans:
                if isinstance(a, dict):
                    at = a.get("answerText", "")
                    length_answer.append(len(at) if isinstance(at, str) else 0)

emit(f"- Tong so dong doc: {total_lines:,}")
emit(f"- So dong JSON hong: {bad_lines:,}")
emit(f"- So asin khac biet (tong): {len(asin_source_count):,}")
emit(f"- So category (tong): {sum(len(v) for v in category_stats.values())}")
emit("")

val_asins = {a for a, c in asin_source_count.items() if c.get("val", 0) > 0}
test_asins = {a for a, c in asin_source_count.items() if c.get("test", 0) > 0}
common_asins = sorted(val_asins & test_asins)
emit(f"- So asin co tai val: {len(val_asins):,}")
emit(f"- So asin co tai test: {len(test_asins):,}")
emit(f"- So asin chung val & test: {len(common_asins):,}")
emit("")

cross_pairs = [k for k, v in pair_source_count.items() if v.get("val", 0) > 0 and v.get("test", 0) > 0]
intra_dup = []
for k, v in pair_source_count.items():
    for src, c in v.items():
        if c > 1:
            intra_dup.append((k[0], k[1], src))
intra_dup_selected = intra_dup[:10]
emit(f"- So cap (asin, questionText_chuan) xuan file (val & test): {len(cross_pairs):,}")
emit(f"- So cap (asin, questionText_chuan) lap noi bo (duoc chon toi da 10): {len(intra_dup_selected)}")
emit("")

emit("=" * 70)
emit("PASS 2: Lay mau asin chung va luu dong thuoc cap (asin, questionText_chuan) duoc chon")
emit("=" * 70)

cross_set = set(cross_pairs)
intra_selected_set = set(intra_dup_selected)

asin_sample = {}
pair_lines = defaultdict(list)
common_asins_set = set(common_asins)

with open(MERGED_V2, "r", encoding="utf-8") as f:
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
        source = obj.get("nguon_goc", "?")
        asin = obj.get("asin", "")
        qnorm = norm_qtext(obj.get("questionText", ""))
        qid = obj.get("qid")
        uid = f"{source}-{qid}"

        key = (asin, qnorm)
        if key in cross_set or (asin, qnorm, source) in intra_selected_set:
            pair_lines[key].append((source, uid, obj))

        if asin in common_asins_set and asin not in asin_sample:
            asin_sample[asin] = obj

emit(f"- So asin chung co lay mau: {len(asin_sample):,}")
emit(f"- So cap (asin, questionText_chuan) co luu dong: {len(pair_lines):,}")
emit("")

emit("=" * 70)
emit("PASS 3: Tinh co tai lieu theo asin (review_snippets, chi luu hash doan)")
emit("=" * 70)

asin_docs = defaultdict(lambda: {"hashes": {}, "tong": 0, "so_doan": 0, "category": ""})

with open(MERGED_V2, "r", encoding="utf-8") as f:
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
        category = obj.get("category", "")
        rec = asin_docs[asin]
        rec["category"] = category
        hashes = rec["hashes"]
        for seg in obj.get("review_snippets", []):
            ns = norm_segment(seg)
            h = seg_hash(ns)
            if h not in hashes:
                hashes[h] = len(ns)
                rec["so_doan"] += 1
                rec["tong"] += len(ns)

emit(f"- So asin co du lieu review_snippets: {len(asin_docs):,}")
emit("")

emit("=" * 70)
emit("BAO CAO THONG KE amazonqa_merged_v2.jsonl")
emit("=" * 70)

emit("")
emit("1. 15 ASIN CHUNG VAL & TEST")
emit("-" * 70)
emit(f"{'ASIN (repr)':<24} {'val':>8} {'test':>8} {'tong':>8}")
for asin in common_asins[:15]:
    c = asin_source_count[asin]
    emit(f"{repr(asin):<24} {c.get('val',0):>8,} {c.get('test',0):>8,} {c.get('val',0)+c.get('test',0):>8,}")

emit("")
emit("1b. Toi da 5 vi du asin chung + moi khoa cua dòng dau tien (gia tri cat ngan):")
shown = 0
for asin in common_asins[:5]:
    obj = asin_sample.get(asin)
    if obj is None:
        continue
    shown += 1
    emit(f"  Vi du {shown}: asin = {repr(asin)}")
    for k in sorted(obj.keys(), key=str):
        emit(f"    {k} = {truncate(obj[k], 150)}")
emit("")

emit("2. CAP (ASIN, QUESTIONTEXT_CHUAN) XUAN FILE VA LAP NOI BO")
emit("-" * 70)
emit(f"2a. So cap xuan file (val & test): {len(cross_pairs):,}")
emit("   UID cua tung dong trong cac cap xuan file (toan bo 115 cap):")
count_shown = 0
for key in cross_pairs:
    rows = pair_lines.get(key, [])
    uids = [uid for _src, uid, _o in rows]
    emit(f"    {key[0]!r} | qtext_chuan={key[1]!r} -> uid: {', '.join(uids)}")
    count_shown += 1
emit(f"   (da in {count_shown} cap xuan file)")

emit("")
emit("2b. Toi da 10 cap lap noi bo (val hoac test):")
for (asin, qnorm, src) in intra_dup_selected:
    rows = pair_lines.get((asin, qnorm), [])
    rows_src = [r for r in rows if r[0] == src]
    uids = [uid for _s, uid, _o in rows_src]
    objs = [o for _s, _u, o in rows_src]
    dks = keys_diff(objs) if len(objs) > 1 else []
    emit(f"    source={src} | asin={repr(asin)} | qtext_chuan={qnorm!r}")
    emit(f"      uid: {', '.join(uids)}")
    if dks:
        emit(f"      khoa co gia tri khac nhau: {', '.join(dks)}")
    else:
        emit(f"      khoa co gia tri khac nhau: (khong - cac dong nhan nhau)")
emit("")

emit("3. THEO CATEGORY (so cau hoi, so asin, so cau hoi/asin trung binh va trung vi)")
emit("-" * 70)
all_categories = sorted(
    {cat for src in category_stats for cat in category_stats[src]}
)
emit(f"{'category':<32} {'src':<6} {'cau hoi':>9} {'asin':>8} {'tb':>9} {'trung vi':>9}")
for cat in all_categories:
    for src in ["val", "test"]:
        cs = category_stats[src].get(cat)
        if cs is None:
            continue
        n_q = cs["count"]
        n_a = len(cs["asins"])
        avg = n_q / n_a if n_a else 0
        per_asin = sorted(category_asin_count[src][cat].values())
        med = statistics.median(per_asin) if per_asin else 0
        emit(f"{cat:<32} {src:<6} {n_q:>9,} {n_a:>8,} {avg:>9.2f} {med:>9.1f}")
emit("")
emit("   (Lưu ý: trung vị số câu hỏi/asin yêu cầu lưu lượt count/asin; trung bình đã in ở cột 'tb'.)")
emit("")

emit("4. CO TAI LIEU THEO ASIN (toan bo file va theo category)")
emit("-" * 70)

def stats_of(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return dict(min=0, median=0, p90=0, max=0)
    return dict(
        min=s[0],
        median=statistics.median(s),
        p90=percentile(s, 90),
        max=s[-1],
    )

all_so_doan = [rec["so_doan"] for rec in asin_docs.values()]
all_tong = [rec["tong"] for rec in asin_docs.values()]
st_sd = stats_of(all_so_doan)
st_tong = stats_of(all_tong)
emit("4a. Toan bo file (tat ca asin):")
emit(f"    So asin: {len(all_so_doan):,}")
emit(f"    So doan khac biet: min={st_sd['min']}, trung vi={st_sd['median']:.1f}, p90={st_sd['p90']:.1f}, max={st_sd['max']}")
emit(f"    Tong ky tu: min={st_tong['min']}, trung vi={st_tong['median']:.1f}, p90={st_tong['p90']:.1f}, max={st_tong['max']}")
for thr in [3000, 5000, 10000]:
    cnt = sum(1 for t in all_tong if t >= thr)
    emit(f"    So asin co tong >= {thr:,} ky tu: {cnt:,}")
emit("")

emit("4b. Theo category (so asin, so doan/tong ky tu - min, trung vi, p90, max; asin >= 3000/5000/10000):")
emit(f"{'category':<32} {'asin':>8} {'sd_min':>7} {'sd_med':>8} {'sd_p90':>8} {'sd_max':>8} {'tg_min':>8} {'tg_med':>9} {'tg_p90':>9} {'tg_max':>9}")
for cat in all_categories:
    sd_vals = []
    tong_vals = []
    cnt3k = cnt5k = cnt10k = 0
    for asin, rec in asin_docs.items():
        if rec["category"] == cat:
            sd_vals.append(rec["so_doan"])
            tong_vals.append(rec["tong"])
            if rec["tong"] >= 3000:
                cnt3k += 1
            if rec["tong"] >= 5000:
                cnt5k += 1
            if rec["tong"] >= 10000:
                cnt10k += 1
    st_sd = stats_of(sd_vals)
    st_tong = stats_of(tong_vals)
    emit(
        f"{cat:<32} {len(sd_vals):>8,} "
        f"{st_sd['min']:>7} {st_sd['median']:>8.1f} {st_sd['p90']:>8.1f} {st_sd['max']:>8} "
        f"{st_tong['min']:>8} {st_tong['median']:>9.1f} {st_tong['p90']:>9.1f} {st_tong['max']:>9} "
        f"(>=3k:{cnt3k:,} >=5k:{cnt5k:,} >=10k:{cnt10k:,})"
    )
emit("")

emit("5. IS_ANSWERABLE, CAU CO >=1 ANSWER, PHAN BO DO DAI")
emit("-" * 70)

emit("5a. is_answerable theo category (val / test):")
emit(f"{'category':<32} {'src':<6} {'ia=0':>8} {'ia=1':>8} {'khac/missing':>12}")
for cat in all_categories:
    for src in ["val", "test"]:
        ctr = is_ans_cat[src].get(cat, Counter())
        n0 = ctr.get("0", 0)
        n1 = ctr.get("1", 0)
        other = sum(v for k, v in ctr.items() if k not in ("0", "1"))
        emit(f"{cat:<32} {src:<6} {n0:>8,} {n1:>8,} {other:>12,}")
emit("")

emit("5b. is_answerable theo questionType (val / test):")
emit(f"{'questionType':<20} {'src':<6} {'ia=0':>8} {'ia=1':>8} {'khac/missing':>12}")
all_qtypes = sorted({qt for src in is_ans_qt for qt in is_ans_qt[src]})
for qt in all_qtypes:
    for src in ["val", "test"]:
        ctr = is_ans_qt[src].get(qt, Counter())
        n0 = ctr.get("0", 0)
        n1 = ctr.get("1", 0)
        other = sum(v for k, v in ctr.items() if k not in ("0", "1"))
        emit(f"{qt:<20} {src:<6} {n0:>8,} {n1:>8,} {other:>12,}")
emit("")

emit("5c. So cau co >= 1 answer:")
for src in ["val", "test"]:
    emit(f"    {src}: {count_answer_ge1[src]:,} cau")
emit("")

emit("5d. Phan bo do dai questionText (ky tu, toan bo file):")
sq = sorted(length_qtext)
st_q = stats_of(sq)
emit(f"    So cau: {len(sq):,}, min={st_q['min']}, trung vi={st_q['median']:.1f}, p90={st_q['p90']:.1f}, max={st_q['max']}, trung binh={statistics.mean(sq):.1f}")
emit("")

emit("5e. Phan bo do dai doan tra loi (answerText, ky tu, toan bo file):")
sa = sorted(length_answer)
st_a = stats_of(sa)
emit(f"    So doan tra loi: {len(sa):,}, min={st_a['min']}, trung vi={st_a['median']:.1f}, p90={st_a['p90']:.1f}, max={st_a['max']}, trung binh={statistics.mean(sa):.1f}")
emit("")

emit("=" * 70)
emit("KET THUC BAO CAO THONG KE V2")
emit("=" * 70)

text = "\n".join(OUT) + "\n"
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(text, encoding="utf-8")
emit(f"\nDa ghi bao cao vao: {REPORT}")
