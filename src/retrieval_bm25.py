"""Module truy xuất cơ sở bằng BM25 (Lexical Baseline Retrieval) trên tập chunks."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.verify_no_labels import BM25TinhDiem, bo_stopword, chuan_hoa, tokenize

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent


def doc_danh_sach_jsonl(path: Path) -> Tuple[List[Dict], int]:
    """Đọc file JSONL, bỏ qua dòng trống và đếm dòng JSON hỏng."""
    items: List[Dict] = []
    n_corrupt = 0
    if not path.exists():
        return items, 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_corrupt += 1
                continue
            if isinstance(obj, dict):
                items.append(obj)
            else:
                n_corrupt += 1
    return items, n_corrupt


def xac_dinh_ground_truth_chunks(
    question_rec: Dict,
    doc_chunks: Sequence[Dict],
    min_overlap_tokens: int = 12,
) -> Set[str]:
    """Xác định tập chunk_id chứa bằng chứng của câu hỏi trong tài liệu cùng doc_id.

    - Với câu hỏi 'no' (unanswerable_in_doc): trả về tập rỗng.
    - Với câu hỏi 'yes' hoặc 'partial': đối chiếu chunk với các đoạn trong
      review_snippets_clean (nếu nhãn 10 đoạn là yes/partial) và bang_chung_ra_soat
      (nếu có bằng chứng rà soát mở rộng).
    """
    nhan_3 = (question_rec.get("nhan_chot_3_muc") or "").strip().lower()
    if nhan_3 not in ("yes", "partial"):
        return set()

    nhan_10 = (question_rec.get("nhan_tay_10_snippets") or "").strip().lower()
    evidence_texts: List[str] = []

    if nhan_10 in ("yes", "partial"):
        for s in question_rec.get("review_snippets_clean") or []:
            if isinstance(s, str) and s.strip():
                evidence_texts.append(s)

    bc = (question_rec.get("bang_chung_ra_soat") or "").strip()
    if bc and not bc.startswith("Đã rà"):
        bc_clean = bc.rstrip(".").strip()
        if bc_clean:
            evidence_texts.append(bc_clean)

    if not evidence_texts:
        for s in question_rec.get("review_snippets_clean") or []:
            if isinstance(s, str) and s.strip():
                evidence_texts.append(s)

    ev_norm_and_tokens: List[Tuple[str, Set[str]]] = []
    for ev in evidence_texts:
        norm_ev = chuan_hoa(ev)
        toks_ev = set(bo_stopword(tokenize(ev)))
        if norm_ev:
            ev_norm_and_tokens.append((norm_ev, toks_ev))

    matched_chunk_ids: Set[str] = set()
    for ch in doc_chunks:
        cid = ch.get("chunk_id")
        ctext = ch.get("text") or ""
        if not cid or not ctext:
            continue
        norm_c = chuan_hoa(ctext)
        toks_c = set(bo_stopword(tokenize(ctext)))
        for norm_ev, toks_ev in ev_norm_and_tokens:
            probe = norm_ev[:120] if len(norm_ev) > 120 else norm_ev
            if probe and (probe in norm_c or norm_c in norm_ev):
                matched_chunk_ids.add(cid)
                break
            common = len(toks_ev & toks_c)
            threshold = min(min_overlap_tokens, max(4, int(len(toks_ev) * 0.6)))
            if len(toks_ev) > 0 and common >= threshold:
                matched_chunk_ids.add(cid)
                break

    return matched_chunk_ids


def truy_xuat_bm25_top_k(
    bm25: BM25TinhDiem,
    chunks: Sequence[Dict],
    query_text: str,
    top_k: int = 5,
    allowed_indices: Optional[Sequence[int]] = None,
) -> List[Dict]:
    """Xếp hạng BM25 và trả về top_k chunks kèm điểm số và thứ hạng."""
    q_tokens = bo_stopword(tokenize(query_text))
    indices = allowed_indices if allowed_indices is not None else range(len(chunks))

    scored: List[Tuple[float, int]] = []
    for idx in indices:
        score = bm25.diem(q_tokens, idx)
        scored.append((score, idx))

    scored.sort(key=lambda x: (-x[0], x[1]))
    top_items = scored[:top_k]

    results: List[Dict] = []
    for rank_idx, (score, idx) in enumerate(top_items, start=1):
        ch = chunks[idx]
        results.append(
            {
                "rank": rank_idx,
                "score": round(float(score), 4),
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "asin": ch["asin"],
                "chunk_index": ch["chunk_index"],
                "n_words": ch["n_words"],
                "text": ch["text"],
            }
        )
    return results


def tinh_hit_va_mrr(
    ranked_items: Sequence[Dict],
    target_key: str,
    valid_targets: Set[str],
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Tính Hit@k và MRR@max_k dựa trên tập đích valid_targets."""
    out: Dict[str, float] = {}
    max_k = max(ks) if ks else 5

    first_hit_rank: Optional[int] = None
    for item in ranked_items[:max_k]:
        val = item.get(target_key)
        if val in valid_targets:
            first_hit_rank = int(item["rank"])
            break

    for k in ks:
        out[f"hit@{k}"] = 1.0 if (first_hit_rank is not None and first_hit_rank <= k) else 0.0

    out[f"mrr@{max_k}"] = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
    return out


def _trung_binh_metrics(rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    if not rows:
        return {k: 0.0 for k in keys}
    n = len(rows)
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in keys}


def chay_danh_gia_bm25_baseline(
    chunks_path: Path,
    eval_questions_path: Path,
    out_jsonl_path: Path,
    report_path: Path,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict:
    """Chạy truy xuất BM25 cơ sở cho 120 câu hỏi đánh giá ở cả 2 chế độ:
    1. Toàn kho (Global: tìm trên toàn bộ 2.814 chunks của 150 sản phẩm).
    2. Trong phạm vi sản phẩm (ASIN-scoped: lọc theo doc_id của sản phẩm đang xem).
    """
    chunks, n_corrupt_chunks = doc_danh_sach_jsonl(chunks_path)
    questions, n_corrupt_q = doc_danh_sach_jsonl(eval_questions_path)

    corpus_tokens = [bo_stopword(tokenize(c.get("text", ""))) for c in chunks]
    bm25_global = BM25TinhDiem(corpus_tokens, k1=k1, b=b)

    doc_to_indices: Dict[str, List[int]] = {}
    doc_to_chunks: Dict[str, List[Dict]] = {}
    for i, c in enumerate(chunks):
        did = c.get("doc_id", "")
        doc_to_indices.setdefault(did, []).append(i)
        doc_to_chunks.setdefault(did, []).append(c)

    metric_keys = ("hit@1", "hit@3", "hit@5", "mrr@5")
    global_asin_metrics_all: List[Dict[str, float]] = []
    global_ev_metrics_ans: List[Dict[str, float]] = []
    scoped_ev_metrics_ans: List[Dict[str, float]] = []
    scoped_by_tier: Dict[str, List[Dict[str, float]]] = {}

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n_questions_with_gt = 0

    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for q in questions:
            qid = q.get("qid_moi", "")
            doc_id = q.get("doc_id", "")
            asin = q.get("asin", "")
            q_text = q.get("question_text_clean") or q.get("question_text") or ""
            nhan_3 = q.get("nhan_chot_3_muc", "")
            nhan_ct = q.get("nhan_chot_chi_tiet", "")

            same_doc_chunks = doc_to_chunks.get(doc_id, [])
            same_doc_indices = doc_to_indices.get(doc_id, [])

            gt_chunk_ids = xac_dinh_ground_truth_chunks(q, same_doc_chunks)

            top_global = truy_xuat_bm25_top_k(
                bm25=bm25_global,
                chunks=chunks,
                query_text=q_text,
                top_k=top_k,
                allowed_indices=None,
            )
            top_scoped = truy_xuat_bm25_top_k(
                bm25=bm25_global,
                chunks=chunks,
                query_text=q_text,
                top_k=top_k,
                allowed_indices=same_doc_indices,
            )

            m_global_asin = tinh_hit_va_mrr(
                ranked_items=top_global,
                target_key="doc_id",
                valid_targets={doc_id},
            )
            global_asin_metrics_all.append(m_global_asin)

            m_global_ev: Optional[Dict[str, float]] = None
            m_scoped_ev: Optional[Dict[str, float]] = None

            if nhan_3 in ("yes", "partial") and gt_chunk_ids:
                n_questions_with_gt += 1
                m_global_ev = tinh_hit_va_mrr(
                    ranked_items=top_global,
                    target_key="chunk_id",
                    valid_targets=gt_chunk_ids,
                )
                m_scoped_ev = tinh_hit_va_mrr(
                    ranked_items=top_scoped,
                    target_key="chunk_id",
                    valid_targets=gt_chunk_ids,
                )
                global_ev_metrics_ans.append(m_global_ev)
                scoped_ev_metrics_ans.append(m_scoped_ev)
                scoped_by_tier.setdefault(nhan_ct, []).append(m_scoped_ev)

            out_rec = {
                "qid_moi": qid,
                "asin": asin,
                "doc_id": doc_id,
                "question_text_clean": q_text,
                "nhan_chot_3_muc": nhan_3,
                "nhan_chot_chi_tiet": nhan_ct,
                "reference_answer": q.get("reference_answer", ""),
                "n_gt_chunks": len(gt_chunk_ids),
                "gt_chunk_ids": sorted(gt_chunk_ids),
                "metrics_global_asin": m_global_asin,
                "metrics_global_evidence": m_global_ev,
                "metrics_scoped_evidence": m_scoped_ev,
                "top_k_scoped": top_scoped,
                "top_k_global": top_global,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    avg_global_asin = _trung_binh_metrics(global_asin_metrics_all, metric_keys)
    avg_global_ev = _trung_binh_metrics(global_ev_metrics_ans, metric_keys)
    avg_scoped_ev = _trung_binh_metrics(scoped_ev_metrics_ans, metric_keys)
    avg_by_tier = {
        tier: _trung_binh_metrics(rows, metric_keys)
        for tier, rows in sorted(scoped_by_tier.items())
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Truy xuất Cơ sở bằng BM25 (Lexical Baseline Retrieval)",
        "",
        "## 1. Cấu hình thực nghiệm",
        f"- Tập đoạn văn cơ sở: `{chunks_path.name}` (**{len(chunks)} chunks** từ 150 sản phẩm `Electronics`, `fixed_size` 200 từ, overlap 40 từ)",
        f"- Tập câu hỏi đánh giá: `{eval_questions_path.name}` (**{len(questions)} câu hỏi** đã kiểm định nhãn 2 vòng)",
        f"- Thuật toán truy xuất: **Okapi BM25** (`k1={k1}`, `b={b}`, tách từ alphanumeric chữ thường, loại bỏ stopword tiếng Anh)",
        f"- Số đoạn truy xuất (`top_k`): **{top_k}**",
        f"- Số câu hỏi có bằng chứng trong tài liệu (`yes` + `partial` khớp chunk đích): **{n_questions_with_gt} / 91 câu** (29 câu còn lại thuộc nhóm `unanswerable_in_doc` không có đoạn đích trong tài liệu)",
        "",
        "## 2. Kết quả Truy xuất Tổng hợp (Hit@1, Hit@3, Hit@5, MRR@5)",
        "",
        "| Chế độ truy xuất & Chỉ số đo | Số câu tính | Hit@1 | Hit@3 | Hit@5 | MRR@5 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
        f"| **1. Toàn kho (Global — Định vị đúng sản phẩm `ASIN`)** | {len(global_asin_metrics_all)} | {avg_global_asin['hit@1']:.4f} | {avg_global_asin['hit@3']:.4f} | {avg_global_asin['hit@5']:.4f} | {avg_global_asin['mrr@5']:.4f} |",
        f"| **2. Toàn kho (Global — Trúng đúng `chunk` chứa bằng chứng)** | {len(global_ev_metrics_ans)} | {avg_global_ev['hit@1']:.4f} | {avg_global_ev['hit@3']:.4f} | {avg_global_ev['hit@5']:.4f} | {avg_global_ev['mrr@5']:.4f} |",
        f"| **3. Trong trang sản phẩm (ASIN-scoped — Trúng `chunk` bằng chứng)** | {len(scoped_ev_metrics_ans)} | **{avg_scoped_ev['hit@1']:.4f}** | **{avg_scoped_ev['hit@3']:.4f}** | **{avg_scoped_ev['hit@5']:.4f}** | **{avg_scoped_ev['mrr@5']:.4f}** |",
        "",
        "## 3. Phân rã kết quả ASIN-scoped theo nhóm nhãn chi tiết",
        "",
        "| Nhóm nhãn chi tiết (`nhan_chot_chi_tiet`) | Số câu | Hit@1 | Hit@3 | Hit@5 | MRR@5 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier, m in avg_by_tier.items():
        n_t = len(scoped_by_tier[tier])
        lines.append(
            f"| `{tier}` | {n_t} | {m['hit@1']:.4f} | {m['hit@3']:.4f} | {m['hit@5']:.4f} | {m['mrr@5']:.4f} |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "n_questions_with_gt": n_questions_with_gt,
        "n_corrupt_chunks": n_corrupt_chunks,
        "n_corrupt_questions": n_corrupt_q,
        "global_asin": avg_global_asin,
        "global_evidence": avg_global_ev,
        "scoped_evidence": avg_scoped_ev,
        "scoped_by_tier": avg_by_tier,
        "out_jsonl_path": str(out_jsonl_path),
        "report_path": str(report_path),
    }
