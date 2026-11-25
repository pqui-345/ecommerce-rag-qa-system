"""Module truy xuất lai (Hybrid Retrieval: Okapi BM25 + Dense Vector bằng RRF)."""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.chunking_eval import (
    ALL_STRATEGIES,
    METRIC_KEYS,
    _nap_hoac_tinh_embeddings,
    _tong_hop_tu_file_retrieval,
    bootstrap_diff_ci,
    mcnemar_exact_pvalue,
)
from src.retrieval_bm25 import (
    _trung_binh_metrics,
    doc_danh_sach_jsonl,
    tinh_hit_va_mrr,
    xac_dinh_ground_truth_chunks,
)
from src.retrieval_vector import BGEFastEmbedder
from src.verify_no_labels import BM25TinhDiem, bo_stopword, tokenize

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k0: int = 60,
    depth: int = 50,
) -> List[Tuple[str, float]]:
    """Hợp nhất nhiều bảng xếp hạng ID theo công thức Reciprocal Rank Fusion (RRF).

    Hòa điểm: ưu tiên thứ hạng tốt hơn ở bảng xếp hạng cuối (nhánh vector), rồi ID tăng dần.
    """
    scores: Dict[str, float] = {}
    tie_rank: Dict[str, int] = {}
    for r_idx, ranked in enumerate(rankings):
        for pos, item_id in enumerate(ranked[:depth], start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k0 + pos)
            if r_idx == len(rankings) - 1:
                tie_rank[item_id] = pos

    fused = list(scores.items())
    fused.sort(key=lambda x: (-x[1], tie_rank.get(x[0], depth + 1), x[0]))
    return [(cid, round(sc, 6)) for cid, sc in fused]


def xep_hang_vector_numpy(
    q_vec: Sequence[float],
    mat: np.ndarray,
    allowed_indices: Optional[Sequence[int]] = None,
    depth: int = 50,
) -> List[Tuple[int, float]]:
    """Xếp hạng cosine chính xác bằng numpy trên ma trận vector đã chuẩn hóa."""
    if mat.size == 0:
        return []
    q = np.asarray(q_vec, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn > 0.0:
        q = q / qn

    if allowed_indices is not None:
        if not allowed_indices:
            return []
        idx_arr = np.asarray(allowed_indices, dtype=np.int64)
        sub = mat[idx_arr]
        sims = sub @ q
        order = np.argsort(-sims, kind="mergesort")[:depth]
        return [(int(idx_arr[i]), float(sims[i])) for i in order]

    sims = mat @ q
    k = min(depth, len(sims))
    if k <= 0:
        return []
    order = np.argsort(-sims, kind="mergesort")[:k]
    return [(int(i), float(sims[i])) for i in order]


def gop_ve_don_vi_cha(
    ranked_child_ids: Sequence[str],
    child_to_parent: Dict[str, str],
    limit: int = 50,
) -> List[str]:
    """Khử trùng danh sách đoạn con về đoạn cha, giữ thứ hạng xuất hiện đầu tiên."""
    seen: Set[str] = set()
    out: List[str] = []
    for cid in ranked_child_ids:
        pid = child_to_parent.get(cid, cid)
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
            if len(out) >= limit:
                break
    return out


def _chuan_hoa_ma_tran(vecs: Sequence[Sequence[float]]) -> np.ndarray:
    if not vecs:
        return np.empty((0, 0), dtype=np.float32)
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _xay_dung_top_k_items(
    fused_ids: Sequence[Tuple[str, float]],
    parent_by_id: Dict[str, Dict],
    bm25_ranks: Dict[str, int],
    vec_ranks: Dict[str, int],
    top_k: int,
) -> List[Dict]:
    items: List[Dict] = []
    for cid, rrf_sc in fused_ids:
        ch = parent_by_id.get(cid)
        if ch is None:
            continue
        items.append(
            {
                "rank": len(items) + 1,
                "score": rrf_sc,
                "rrf_score": rrf_sc,
                "rank_bm25": bm25_ranks.get(cid),
                "rank_vector": vec_ranks.get(cid),
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "asin": ch["asin"],
                "chunk_index": ch["chunk_index"],
                "n_words": ch["n_words"],
                "text": ch["text"],
            }
        )
        if len(items) >= top_k:
            break
    return items


def danh_gia_hybrid_chien_luoc(
    strategy: str,
    chunks_path: Path,
    eval_questions_path: Path,
    out_jsonl_path: Path,
    children_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    embed_docs_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    embed_queries_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    k0: int = 60,
    depth: int = 50,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    k0_sweep: Sequence[int] = (10, 30, 60, 100),
    existing_vector_jsonl: Optional[Path] = None,
) -> Dict:
    """Chạy truy xuất lai BM25 + Vector bằng RRF cho một chiến lược phân đoạn."""
    parent_chunks, _ = doc_danh_sach_jsonl(chunks_path)
    questions, _ = doc_danh_sach_jsonl(eval_questions_path)

    is_pc = strategy == "parent_child" and children_path is not None and children_path.exists()
    index_chunks, _ = doc_danh_sach_jsonl(children_path) if is_pc else (parent_chunks, 0)
    parent_by_id = {c["chunk_id"]: c for c in parent_chunks}
    child_to_parent = {c["chunk_id"]: c.get("parent_id", c["chunk_id"]) for c in index_chunks}

    if embed_docs_fn is None or embed_queries_fn is None:
        embedder = BGEFastEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=32)
        embed_docs_fn = embedder.embed_documents
        embed_queries_fn = embedder.embed_queries

    cache_file = (cache_dir / f"emb_dev_{strategy}.npy") if cache_dir else None
    all_texts = [str(c.get("text", "")) for c in index_chunks]
    all_vecs = _nap_hoac_tinh_embeddings(all_texts, cache_file, embed_docs_fn)
    mat = _chuan_hoa_ma_tran(all_vecs)

    q_texts = [(q.get("question_text_clean") or q.get("question_text") or "") for q in questions]
    q_cache = (cache_dir / "emb_dev_queries_120.npy") if cache_dir else None
    q_vecs = _nap_hoac_tinh_embeddings(q_texts, q_cache, embed_queries_fn)

    corpus_tokens = [bo_stopword(tokenize(c.get("text", ""))) for c in index_chunks]
    bm25 = BM25TinhDiem(corpus_tokens, k1=k1, b=b)

    doc_to_idx: Dict[str, List[int]] = {}
    for i, c in enumerate(index_chunks):
        doc_to_idx.setdefault(c.get("doc_id", ""), []).append(i)

    doc_to_parents: Dict[str, List[Dict]] = {}
    for c in parent_chunks:
        doc_to_parents.setdefault(c.get("doc_id", ""), []).append(c)

    existing_vec_top5: Dict[str, List[str]] = {}
    if existing_vector_jsonl is not None and existing_vector_jsonl.exists():
        old_rows, _ = doc_danh_sach_jsonl(existing_vector_jsonl)
        for r in old_rows:
            existing_vec_top5[r["qid_moi"]] = [c["chunk_id"] for c in (r.get("top_k_scoped") or [])[:top_k]]

    search_depth = depth * 4 if is_pc else depth
    sweep_scoped_metrics: Dict[int, List[Dict[str, float]]] = {kv: [] for kv in k0_sweep}
    sweep_global_asin: Dict[int, List[Dict[str, float]]] = {kv: [] for kv in k0_sweep}
    numpy_vs_chroma_overlap: List[float] = []

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for q, q_text, q_vec in zip(questions, q_texts, q_vecs):
            qid = q.get("qid_moi", "")
            doc_id = q.get("doc_id", "")
            asin = q.get("asin", "")
            nhan_3 = q.get("nhan_chot_3_muc", "")
            nhan_ct = q.get("nhan_chot_chi_tiet", "")

            same_doc_parents = doc_to_parents.get(doc_id, [])
            gt_chunk_ids = xac_dinh_ground_truth_chunks(q, same_doc_parents)

            q_tokens = bo_stopword(tokenize(q_text))

            bm_g_scored = [(bm25.diem(q_tokens, i), i) for i in range(len(index_chunks))]
            bm_g_scored.sort(key=lambda x: (-x[0], x[1]))
            bm_g_ids_raw = [index_chunks[i]["chunk_id"] for _, i in bm_g_scored[:search_depth]]

            vec_g_scored = xep_hang_vector_numpy(q_vec, mat, None, depth=search_depth)
            vec_g_ids_raw = [index_chunks[i]["chunk_id"] for i, _ in vec_g_scored]

            scoped_indices = doc_to_idx.get(doc_id, [])
            bm_s_scored = [(bm25.diem(q_tokens, i), i) for i in scoped_indices]
            bm_s_scored.sort(key=lambda x: (-x[0], x[1]))
            bm_s_ids_raw = [index_chunks[i]["chunk_id"] for _, i in bm_s_scored[:search_depth]]

            vec_s_scored = xep_hang_vector_numpy(q_vec, mat, scoped_indices, depth=search_depth)
            vec_s_ids_raw = [index_chunks[i]["chunk_id"] for i, _ in vec_s_scored]

            if is_pc:
                bm_g_ids = gop_ve_don_vi_cha(bm_g_ids_raw, child_to_parent, limit=depth)
                vec_g_ids = gop_ve_don_vi_cha(vec_g_ids_raw, child_to_parent, limit=depth)
                bm_s_ids = gop_ve_don_vi_cha(bm_s_ids_raw, child_to_parent, limit=depth)
                vec_s_ids = gop_ve_don_vi_cha(vec_s_ids_raw, child_to_parent, limit=depth)
            else:
                bm_g_ids = bm_g_ids_raw[:depth]
                vec_g_ids = vec_g_ids_raw[:depth]
                bm_s_ids = bm_s_ids_raw[:depth]
                vec_s_ids = vec_s_ids_raw[:depth]

            if qid in existing_vec_top5 and existing_vec_top5[qid]:
                old_set = set(existing_vec_top5[qid])
                new_set = set(vec_s_ids[:top_k])
                numpy_vs_chroma_overlap.append(len(old_set & new_set) / max(1, len(old_set)))

            bm_g_rank_map = {cid: pos for pos, cid in enumerate(bm_g_ids, start=1)}
            vec_g_rank_map = {cid: pos for pos, cid in enumerate(vec_g_ids, start=1)}
            bm_s_rank_map = {cid: pos for pos, cid in enumerate(bm_s_ids, start=1)}
            vec_s_rank_map = {cid: pos for pos, cid in enumerate(vec_s_ids, start=1)}

            fused_g = rrf_fuse([bm_g_ids, vec_g_ids], k0=k0, depth=depth)
            fused_s = rrf_fuse([bm_s_ids, vec_s_ids], k0=k0, depth=depth)

            top_global = _xay_dung_top_k_items(fused_g, parent_by_id, bm_g_rank_map, vec_g_rank_map, top_k)
            top_scoped = _xay_dung_top_k_items(fused_s, parent_by_id, bm_s_rank_map, vec_s_rank_map, top_k)

            m_global_asin = tinh_hit_va_mrr(top_global, "doc_id", {doc_id})
            m_global_ev = None
            m_scoped_ev = None
            if nhan_3 in ("yes", "partial") and gt_chunk_ids:
                m_global_ev = tinh_hit_va_mrr(top_global, "chunk_id", gt_chunk_ids)
                m_scoped_ev = tinh_hit_va_mrr(top_scoped, "chunk_id", gt_chunk_ids)

            for kv in k0_sweep:
                fg_k = rrf_fuse([bm_g_ids, vec_g_ids], k0=kv, depth=depth)
                tg_k = _xay_dung_top_k_items(fg_k, parent_by_id, bm_g_rank_map, vec_g_rank_map, top_k)
                sweep_global_asin[kv].append(tinh_hit_va_mrr(tg_k, "doc_id", {doc_id}))
                if nhan_3 in ("yes", "partial") and gt_chunk_ids:
                    fs_k = rrf_fuse([bm_s_ids, vec_s_ids], k0=kv, depth=depth)
                    ts_k = _xay_dung_top_k_items(fs_k, parent_by_id, bm_s_rank_map, vec_s_rank_map, top_k)
                    sweep_scoped_metrics[kv].append(tinh_hit_va_mrr(ts_k, "chunk_id", gt_chunk_ids))

            out_rec = {
                "qid_moi": qid,
                "asin": asin,
                "doc_id": doc_id,
                "strategy": strategy,
                "method": "hybrid_rrf",
                "k0": k0,
                "depth": depth,
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

    summary = _tong_hop_tu_file_retrieval(strategy, "hybrid_rrf", out_jsonl_path, chunks_path)
    summary["k0_sweep_scoped"] = {
        kv: _trung_binh_metrics(rows, METRIC_KEYS) for kv, rows in sweep_scoped_metrics.items()
    }
    summary["k0_sweep_global_asin"] = {
        kv: _trung_binh_metrics(rows, METRIC_KEYS) for kv, rows in sweep_global_asin.items()
    }
    summary["numpy_vs_chroma_top5_overlap"] = (
        round(sum(numpy_vs_chroma_overlap) * 100.0 / len(numpy_vs_chroma_overlap), 2)
        if numpy_vs_chroma_overlap
        else None
    )
    return summary


def ghi_bao_cao_hybrid_truy_xuat(
    hybrid_results: Sequence[Dict],
    bm25_by_st: Dict[str, Dict],
    vec_by_st: Dict[str, Dict],
    report_path: Path,
    k0: int = 60,
    depth: int = 50,
) -> None:
    """Xuất báo cáo Markdown đối sánh Truy xuất lai RRF với BM25 và Dense Vector."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Đối sánh Truy xuất Lai (Hybrid BM25 + Dense Vector bằng RRF — Hạng mục 2)",
        "",
        "## 1. Cấu hình thực nghiệm",
        f"- Công thức hợp nhất: **Reciprocal Rank Fusion (RRF)** với `k0 = {k0}`, độ sâu mỗi nhánh `depth = {depth}`, `top_k = 5`",
        "- Nhánh Lexical: **Okapi BM25** (`k1=1.5, b=0.75`); Nhánh Dense: **`BAAI/bge-small-en-v1.5`** (Cosine chính xác trên vector đã cache)",
        "- Tập dữ liệu: `docs_dev_clean.jsonl` (150 sản phẩm `Electronics`), `eval_120_dev_labeled.jsonl` (120 câu hỏi)",
        "",
        "## 2. So sánh Truy xuất Toàn kho (`Global ASIN`, 120 câu) & Trong Sản phẩm (`ASIN-scoped`, 91 câu)",
        "",
        "| Chiến lược | Phương pháp | Global ASIN Hit@1 | Global ASIN Hit@5 | Global ASIN MRR@5 | Scoped Hit@1 | Scoped Hit@3 | Scoped Hit@5 | Scoped MRR@5 | Câu khó `recovered` (Hit@5 / MRR@5) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for hr in hybrid_results:
        st = hr["strategy"]
        for label, r in (
            ("BM25", bm25_by_st.get(st)),
            ("Vector (`BGE-small`)", vec_by_st.get(st)),
            ("**Hybrid RRF**", hr),
        ):
            if not r:
                continue
            ga = r["global_asin"]
            sc = r["scoped_evidence"]
            rec = r["scoped_by_tier"].get("recovered_in_full_doc", {k: 0.0 for k in METRIC_KEYS})
            lines.append(
                f"| `{st}` | {label} | {ga['hit@1']:.4f} | {ga['hit@5']:.4f} | {ga['mrr@5']:.4f} | "
                f"{sc['hit@1']:.4f} | {sc['hit@3']:.4f} | {sc['hit@5']:.4f} | {sc['mrr@5']:.4f} | "
                f"{rec['hit@5']:.4f} / {rec['mrr@5']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 3. Kiểm định Ý nghĩa Thống kê: Hybrid RRF vs. Dense Vector (cùng chiến lược, 91 câu ASIN-scoped)",
            "",
            "| Chiến lược | $\\Delta$ Global ASIN Hit@5 | $\\Delta$ Scoped Hit@5 | McNemar Scoped Hit@5 ($b / c$, $p$-value) | $\\Delta$ Scoped MRR@5 | Bootstrap 95% CI $\\Delta$ MRR@5 | Trùng Top-5 Numpy vs Chroma |",
            "| :--- | ---: | ---: | :--- | ---: | :--- | ---: |",
        ]
    )
    for hr in hybrid_results:
        st = hr["strategy"]
        vr = vec_by_st.get(st)
        if not vr:
            continue
        qb = vr["scoped_by_qid"]
        qn = hr["scoped_by_qid"]
        common = sorted(set(qb.keys()) & set(qn.keys()))
        h5_b = [qb[q]["hit@5"] for q in common]
        h5_n = [qn[q]["hit@5"] for q in common]
        mrr_b = [qb[q]["mrr@5"] for q in common]
        mrr_n = [qn[q]["mrr@5"] for q in common]
        mc = mcnemar_exact_pvalue(h5_b, h5_n)
        bt = bootstrap_diff_ci(mrr_b, mrr_n)
        d_ga = hr["global_asin"]["hit@5"] - vr["global_asin"]["hit@5"]
        d_sc = hr["scoped_evidence"]["hit@5"] - vr["scoped_evidence"]["hit@5"]
        ov = hr.get("numpy_vs_chroma_top5_overlap")
        ov_str = f"{ov:.2f}%" if ov is not None else "-"
        lines.append(
            f"| `{st}` | {d_ga:+.4f} | {d_sc:+.4f} | "
            f"`b={mc['b_only_base']}, c={mc['c_only_new']}` ($p={mc['p_value']:.4f}$) | "
            f"{bt['diff_mean']:+.4f} | `[{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]` | {ov_str} |"
        )

    lines.extend(
        [
            "",
            "## 4. Phân tích độ nhạy tham số $k_0 \\in \\{10, 30, 60, 100\\}$ (Hybrid RRF)",
            "",
            "| Chiến lược | $k_0 = 10$ (Global / Scoped Hit@5 / MRR@5) | $k_0 = 30$ | $k_0 = 60$ (Chuẩn) | $k_0 = 100$ |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for hr in hybrid_results:
        st = hr["strategy"]
        cols = []
        for kv in (10, 30, 60, 100):
            sg = hr.get("k0_sweep_global_asin", {}).get(kv, {})
            ss = hr.get("k0_sweep_scoped", {}).get(kv, {})
            if sg and ss:
                cols.append(f"{sg['hit@5']:.4f} / {ss['hit@5']:.4f} / {ss['mrr@5']:.4f}")
            else:
                cols.append("-")
        lines.append(f"| `{st}` | {cols[0]} | {cols[1]} | **{cols[2]}** | {cols[3]} |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def chay_hybrid_6_chien_luoc(
    processed_dir: Path,
    eval_questions_path: Path,
    report_path: Path,
    strategies: Sequence[str] = ALL_STRATEGIES,
    k0: int = 60,
    depth: int = 50,
    top_k: int = 5,
    embed_docs_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    embed_queries_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
) -> Dict:
    """Thực thi Truy xuất lai RRF trên các chiến lược phân đoạn và xuất báo cáo đối sánh."""
    cache_dir = processed_dir / "emb_cache"
    if embed_docs_fn is None or embed_queries_fn is None:
        embedder = BGEFastEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=32)
        embed_docs_fn = embedder.embed_documents
        embed_queries_fn = embedder.embed_queries

    hybrid_results: List[Dict] = []
    bm25_by_st: Dict[str, Dict] = {}
    vec_by_st: Dict[str, Dict] = {}

    for st in strategies:
        chunks_path = processed_dir / f"chunks_dev_{st}.jsonl"
        children_path = (
            processed_dir / "chunks_dev_parent_child_children.jsonl"
            if st == "parent_child"
            else None
        )
        bm25_path = processed_dir / f"retrieval_bm25_dev_{st}.jsonl"
        vec_path = processed_dir / f"retrieval_vector_dev_{st}.jsonl"
        out_path = processed_dir / f"retrieval_hybrid_dev_{st}.jsonl"

        if bm25_path.exists():
            bm25_by_st[st] = _tong_hop_tu_file_retrieval(st, "bm25", bm25_path, chunks_path)
        if vec_path.exists():
            vec_by_st[st] = _tong_hop_tu_file_retrieval(st, "vector", vec_path, chunks_path)

        res = danh_gia_hybrid_chien_luoc(
            strategy=st,
            chunks_path=chunks_path,
            eval_questions_path=eval_questions_path,
            out_jsonl_path=out_path,
            children_path=children_path,
            cache_dir=cache_dir,
            embed_docs_fn=embed_docs_fn,
            embed_queries_fn=embed_queries_fn,
            k0=k0,
            depth=depth,
            top_k=top_k,
            existing_vector_jsonl=vec_path if vec_path.exists() else None,
        )
        hybrid_results.append(res)
        print(
            f"[Hybrid {st}] Global ASIN Hit@5={res['global_asin']['hit@5']:.4f} "
            f"Scoped Hit@5={res['scoped_evidence']['hit@5']:.4f} "
            f"MRR@5={res['scoped_evidence']['mrr@5']:.4f} "
            f"NumpyVsChroma={res['numpy_vs_chroma_top5_overlap']}%",
            flush=True,
        )

    ghi_bao_cao_hybrid_truy_xuat(hybrid_results, bm25_by_st, vec_by_st, report_path, k0=k0, depth=depth)
    return {
        "hybrid": hybrid_results,
        "report_path": str(report_path),
    }
