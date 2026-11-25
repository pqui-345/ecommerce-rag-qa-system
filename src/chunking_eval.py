"""Đánh giá và đối sánh truy xuất (BM25 & Dense Vector) trên 6 chiến lược phân đoạn (RQ1)."""

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.retrieval_bm25 import (
    _trung_binh_metrics,
    doc_danh_sach_jsonl,
    tinh_hit_va_mrr,
    xac_dinh_ground_truth_chunks,
)
from src.retrieval_vector import (
    BGEFastEmbedder,
    tao_hoac_nap_chroma_collection,
    truy_xuat_chroma_top_k,
)
from src.verify_no_labels import BM25TinhDiem, bo_stopword, tokenize

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ALL_STRATEGIES = ("fixed", "recursive", "sentence", "structure", "semantic", "parent_child")
METRIC_KEYS = ("hit@1", "hit@3", "hit@5", "mrr@5")


def mcnemar_exact_pvalue(hits_a: Sequence[float], hits_b: Sequence[float]) -> Dict[str, float]:
    """Kiểm định McNemar hai phía chính xác (phân phối nhị thức) trên cặp kết quả nhị phân."""
    b = 0
    c = 0
    for xa, xb in zip(hits_a, hits_b):
        ia = 1 if xa > 0.5 else 0
        ib = 1 if xb > 0.5 else 0
        if ia == 1 and ib == 0:
            b += 1
        elif ia == 0 and ib == 1:
            c += 1
    n = b + c
    if n == 0:
        return {"b_only_base": 0, "c_only_new": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    p_val = min(1.0, 2.0 * tail)
    return {"b_only_base": b, "c_only_new": c, "p_value": round(p_val, 4)}


def bootstrap_diff_ci(
    vals_base: Sequence[float],
    vals_new: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Khoảng tin cậy 95% bootstrap theo cặp cho chênh lệch trung bình (new - base)."""
    n = len(vals_base)
    if n == 0 or len(vals_new) != n:
        return {"diff_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    diffs = [eb - ea for ea, eb in zip(vals_base, vals_new)]
    obs = sum(diffs) / n
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {
        "diff_mean": round(obs, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
    }


def _dedup_children_to_parents(
    ranked_children: Sequence[Dict],
    parent_by_id: Dict[str, Dict],
    top_k: int = 5,
) -> List[Dict]:
    """Ánh xạ danh sách đoạn con xếp hạng cao về đoạn cha duy nhất theo thứ tự điểm cao nhất."""
    seen_parents: Set[str] = set()
    out: List[Dict] = []
    for item in ranked_children:
        pid = item.get("parent_id") or item.get("chunk_id")
        if not pid or pid in seen_parents:
            continue
        p_chunk = parent_by_id.get(pid)
        if p_chunk is None:
            continue
        seen_parents.add(pid)
        out.append(
            {
                "rank": len(out) + 1,
                "score": item["score"],
                "chunk_id": p_chunk["chunk_id"],
                "doc_id": p_chunk["doc_id"],
                "asin": p_chunk["asin"],
                "chunk_index": p_chunk["chunk_index"],
                "n_words": p_chunk["n_words"],
                "text": p_chunk["text"],
                "matched_child_id": item["chunk_id"],
            }
        )
        if len(out) >= top_k:
            break
    return out


def danh_gia_bm25_chien_luoc(
    strategy: str,
    chunks_path: Path,
    eval_questions_path: Path,
    out_jsonl_path: Path,
    children_path: Optional[Path] = None,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    reuse_existing: bool = True,
) -> Dict:
    """Chạy hoặc nạp kết quả đánh giá BM25 cho một chiến lược phân đoạn."""
    if reuse_existing and out_jsonl_path.exists():
        return _tong_hop_tu_file_retrieval(strategy, "bm25", out_jsonl_path, chunks_path)

    parent_chunks, _ = doc_danh_sach_jsonl(chunks_path)
    questions, _ = doc_danh_sach_jsonl(eval_questions_path)

    is_pc = strategy == "parent_child" and children_path is not None and children_path.exists()
    index_chunks, _ = doc_danh_sach_jsonl(children_path) if is_pc else (parent_chunks, 0)
    parent_by_id = {c["chunk_id"]: c for c in parent_chunks}

    corpus_tokens = [bo_stopword(tokenize(c.get("text", ""))) for c in index_chunks]
    bm25 = BM25TinhDiem(corpus_tokens, k1=k1, b=b)

    doc_to_idx: Dict[str, List[int]] = {}
    for i, c in enumerate(index_chunks):
        doc_to_idx.setdefault(c.get("doc_id", ""), []).append(i)

    doc_to_parents: Dict[str, List[Dict]] = {}
    for c in parent_chunks:
        doc_to_parents.setdefault(c.get("doc_id", ""), []).append(c)

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    search_depth = top_k * 6 if is_pc else top_k

    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for q in questions:
            qid = q.get("qid_moi", "")
            doc_id = q.get("doc_id", "")
            asin = q.get("asin", "")
            q_text = q.get("question_text_clean") or q.get("question_text") or ""
            nhan_3 = q.get("nhan_chot_3_muc", "")
            nhan_ct = q.get("nhan_chot_chi_tiet", "")

            same_doc_parents = doc_to_parents.get(doc_id, [])
            gt_chunk_ids = xac_dinh_ground_truth_chunks(q, same_doc_parents)

            q_tokens = bo_stopword(tokenize(q_text))
            scored_global = [(bm25.diem(q_tokens, i), i) for i in range(len(index_chunks))]
            scored_global.sort(key=lambda x: (-x[0], x[1]))
            raw_global = [
                {
                    "rank": r,
                    "score": round(float(sc), 4),
                    "chunk_id": index_chunks[i]["chunk_id"],
                    "parent_id": index_chunks[i].get("parent_id", index_chunks[i]["chunk_id"]),
                    "doc_id": index_chunks[i]["doc_id"],
                    "asin": index_chunks[i]["asin"],
                    "chunk_index": index_chunks[i]["chunk_index"],
                    "n_words": index_chunks[i]["n_words"],
                    "text": index_chunks[i]["text"],
                }
                for r, (sc, i) in enumerate(scored_global[:search_depth], start=1)
            ]

            scoped_indices = doc_to_idx.get(doc_id, [])
            scored_scoped = [(bm25.diem(q_tokens, i), i) for i in scoped_indices]
            scored_scoped.sort(key=lambda x: (-x[0], x[1]))
            raw_scoped = [
                {
                    "rank": r,
                    "score": round(float(sc), 4),
                    "chunk_id": index_chunks[i]["chunk_id"],
                    "parent_id": index_chunks[i].get("parent_id", index_chunks[i]["chunk_id"]),
                    "doc_id": index_chunks[i]["doc_id"],
                    "asin": index_chunks[i]["asin"],
                    "chunk_index": index_chunks[i]["chunk_index"],
                    "n_words": index_chunks[i]["n_words"],
                    "text": index_chunks[i]["text"],
                }
                for r, (sc, i) in enumerate(scored_scoped[:search_depth], start=1)
            ]

            if is_pc:
                top_global = _dedup_children_to_parents(raw_global, parent_by_id, top_k=top_k)
                top_scoped = _dedup_children_to_parents(raw_scoped, parent_by_id, top_k=top_k)
            else:
                for it in raw_global:
                    it.pop("parent_id", None)
                for it in raw_scoped:
                    it.pop("parent_id", None)
                top_global = raw_global[:top_k]
                top_scoped = raw_scoped[:top_k]

            m_global_asin = tinh_hit_va_mrr(top_global, "doc_id", {doc_id})
            m_global_ev = None
            m_scoped_ev = None
            if nhan_3 in ("yes", "partial") and gt_chunk_ids:
                m_global_ev = tinh_hit_va_mrr(top_global, "chunk_id", gt_chunk_ids)
                m_scoped_ev = tinh_hit_va_mrr(top_scoped, "chunk_id", gt_chunk_ids)

            out_rec = {
                "qid_moi": qid,
                "asin": asin,
                "doc_id": doc_id,
                "strategy": strategy,
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

    return _tong_hop_tu_file_retrieval(strategy, "bm25", out_jsonl_path, chunks_path)


_TEXT_EMB_MEMO: Dict[str, List[float]] = {}


def _nap_hoac_tinh_embeddings(
    texts: Sequence[str],
    cache_path: Optional[Path],
    embed_fn: Callable[[Sequence[str]], List[List[float]]],
) -> List[List[float]]:
    if cache_path is not None and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            vecs_loaded = arr.tolist()
            for t, v in zip(texts, vecs_loaded):
                _TEXT_EMB_MEMO.setdefault(t, v)
            return vecs_loaded
    missing_texts = list({t for t in texts if t not in _TEXT_EMB_MEMO})
    if missing_texts:
        new_vecs = embed_fn(missing_texts)
        for t, v in zip(missing_texts, new_vecs):
            _TEXT_EMB_MEMO[t] = v
    vecs = [_TEXT_EMB_MEMO[t] for t in texts]
    if cache_path is not None and vecs:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.asarray(vecs, dtype=np.float32))
    return vecs


def danh_gia_vector_chien_luoc(
    strategy: str,
    chunks_path: Path,
    eval_questions_path: Path,
    out_jsonl_path: Path,
    children_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    embed_docs_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    embed_queries_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    top_k: int = 5,
    reuse_existing: bool = True,
) -> Dict:
    """Chạy hoặc nạp kết quả đánh giá Dense Vector cho một chiến lược phân đoạn."""
    if reuse_existing and out_jsonl_path.exists():
        return _tong_hop_tu_file_retrieval(strategy, "vector", out_jsonl_path, chunks_path)

    parent_chunks, _ = doc_danh_sach_jsonl(chunks_path)
    questions, _ = doc_danh_sach_jsonl(eval_questions_path)

    is_pc = strategy == "parent_child" and children_path is not None and children_path.exists()
    index_chunks, _ = doc_danh_sach_jsonl(children_path) if is_pc else (parent_chunks, 0)
    parent_by_id = {c["chunk_id"]: c for c in parent_chunks}

    if embed_docs_fn is None or embed_queries_fn is None:
        embedder = BGEFastEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=64)
        embed_docs_fn = embedder.embed_documents
        embed_queries_fn = embedder.embed_queries

    cache_file = (cache_dir / f"emb_dev_{strategy}.npy") if cache_dir else None
    all_texts = [str(c.get("text", "")) for c in index_chunks]
    all_vecs = _nap_hoac_tinh_embeddings(all_texts, cache_file, embed_docs_fn)

    vec_iter = iter(all_vecs)

    def _precomputed_embed(batch_docs: Sequence[str]) -> List[List[float]]:
        return [next(vec_iter) for _ in batch_docs]

    collection = tao_hoac_nap_chroma_collection(
        chunks=index_chunks,
        embed_docs_fn=_precomputed_embed,
        collection_name=f"dev_{strategy}",
        persist_dir=None,
        batch_size=512,
        force_rebuild=True,
    )

    child_to_parent = {c["chunk_id"]: c.get("parent_id", c["chunk_id"]) for c in index_chunks}
    doc_to_parents: Dict[str, List[Dict]] = {}
    for c in parent_chunks:
        doc_to_parents.setdefault(c.get("doc_id", ""), []).append(c)
    doc_to_index_count: Dict[str, int] = {}
    for c in index_chunks:
        did = c.get("doc_id", "")
        doc_to_index_count[did] = doc_to_index_count.get(did, 0) + 1

    q_texts = [(q.get("question_text_clean") or q.get("question_text") or "") for q in questions]
    q_cache = (cache_dir / "emb_dev_queries_120.npy") if cache_dir else None
    q_vecs = _nap_hoac_tinh_embeddings(q_texts, q_cache, embed_queries_fn)

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    search_depth = top_k * 6 if is_pc else top_k

    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for q, q_text, q_vec in zip(questions, q_texts, q_vecs):
            qid = q.get("qid_moi", "")
            doc_id = q.get("doc_id", "")
            asin = q.get("asin", "")
            nhan_3 = q.get("nhan_chot_3_muc", "")
            nhan_ct = q.get("nhan_chot_chi_tiet", "")

            same_doc_parents = doc_to_parents.get(doc_id, [])
            gt_chunk_ids = xac_dinh_ground_truth_chunks(q, same_doc_parents)

            raw_global = truy_xuat_chroma_top_k(
                collection=collection,
                query_vec=q_vec,
                top_k=min(search_depth, len(index_chunks)),
                where_filter=None,
            )
            n_in_doc = doc_to_index_count.get(doc_id, 0)
            raw_scoped = (
                truy_xuat_chroma_top_k(
                    collection=collection,
                    query_vec=q_vec,
                    top_k=min(search_depth, n_in_doc),
                    where_filter={"doc_id": doc_id},
                )
                if n_in_doc > 0
                else []
            )

            if is_pc:
                for it in raw_global:
                    it["parent_id"] = child_to_parent.get(it["chunk_id"], it["chunk_id"])
                for it in raw_scoped:
                    it["parent_id"] = child_to_parent.get(it["chunk_id"], it["chunk_id"])
                top_global = _dedup_children_to_parents(raw_global, parent_by_id, top_k=top_k)
                top_scoped = _dedup_children_to_parents(raw_scoped, parent_by_id, top_k=top_k)
            else:
                top_global = raw_global[:top_k]
                top_scoped = raw_scoped[:top_k]

            m_global_asin = tinh_hit_va_mrr(top_global, "doc_id", {doc_id})
            m_global_ev = None
            m_scoped_ev = None
            if nhan_3 in ("yes", "partial") and gt_chunk_ids:
                m_global_ev = tinh_hit_va_mrr(top_global, "chunk_id", gt_chunk_ids)
                m_scoped_ev = tinh_hit_va_mrr(top_scoped, "chunk_id", gt_chunk_ids)

            out_rec = {
                "qid_moi": qid,
                "asin": asin,
                "doc_id": doc_id,
                "strategy": strategy,
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

    return _tong_hop_tu_file_retrieval(strategy, "vector", out_jsonl_path, chunks_path)


def _tong_hop_tu_file_retrieval(
    strategy: str,
    method: str,
    jsonl_path: Path,
    chunks_path: Path,
) -> Dict:
    rows, _ = doc_danh_sach_jsonl(jsonl_path)
    chunks, _ = doc_danh_sach_jsonl(chunks_path)

    g_asin = [r["metrics_global_asin"] for r in rows if r.get("metrics_global_asin")]
    g_ev = [r["metrics_global_evidence"] for r in rows if r.get("metrics_global_evidence")]
    s_ev = [r["metrics_scoped_evidence"] for r in rows if r.get("metrics_scoped_evidence")]

    by_tier: Dict[str, List[Dict[str, float]]] = {}
    scoped_by_qid: Dict[str, Dict[str, float]] = {}
    top5_words: List[int] = []

    for r in rows:
        tk = r.get("top_k_scoped") or []
        top5_words.append(sum(int(c.get("n_words", 0)) for c in tk[:5]))
        m_sc = r.get("metrics_scoped_evidence")
        if m_sc:
            tier = r.get("nhan_chot_chi_tiet", "")
            by_tier.setdefault(tier, []).append(m_sc)
            scoped_by_qid[r["qid_moi"]] = m_sc

    avg_by_tier = {t: _trung_binh_metrics(v, METRIC_KEYS) for t, v in sorted(by_tier.items())}
    mean_words = round(sum(top5_words) / len(top5_words), 1) if top5_words else 0.0

    return {
        "strategy": strategy,
        "method": method,
        "n_chunks": len(chunks),
        "n_questions": len(rows),
        "n_questions_with_gt": len(s_ev),
        "mean_top5_words": mean_words,
        "global_asin": _trung_binh_metrics(g_asin, METRIC_KEYS),
        "global_evidence": _trung_binh_metrics(g_ev, METRIC_KEYS),
        "scoped_evidence": _trung_binh_metrics(s_ev, METRIC_KEYS),
        "scoped_by_tier": avg_by_tier,
        "scoped_by_qid": scoped_by_qid,
        "out_jsonl_path": str(jsonl_path),
    }


def ghi_bao_cao_so_sanh_truy_xuat(
    bm25_results: Sequence[Dict],
    vector_results: Sequence[Dict],
    report_path: Path,
) -> None:
    """Xuất báo cáo Markdown đối sánh 6 chiến lược phân đoạn trên BM25 và Dense Vector."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Đối sánh Hiệu năng Truy xuất giữa 6 Chiến lược Phân đoạn (RQ1)",
        "",
        "## 1. Cấu hình thực nghiệm đối chứng",
        "- Tập tài liệu: `docs_dev_clean.jsonl` (**150 sản phẩm Electronics**, ngân sách trần **200 từ/đoạn**)",
        "- Tập câu hỏi đánh giá: `eval_120_dev_labeled.jsonl` (**120 câu hỏi** kiểm tay 2 vòng: 27 `yes`, 64 `partial`, 29 `no`)",
        "- Hai nhánh truy xuất độc lập: **Okapi BM25** (`k1=1.5, b=0.75`) và **Dense Vector `BAAI/bge-small-en-v1.5`** (ChromaDB Cosine, `top_k=5`)",
        "",
        "## 2. Kết quả Truy xuất trong Trang Sản phẩm (`ASIN-scoped Evidence`, 91 câu có bằng chứng)",
        "",
        "### 2.1. Nhánh Dense Vector (`BAAI/bge-small-en-v1.5` + `ChromaDB`)",
        "",
        "| Chiến lược | Số chunks | Số câu có GT | Từ ngữ cảnh Top-5 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Nhóm câu khó `recovered` (Hit@5 / MRR@5) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in vector_results:
        sc = r["scoped_evidence"]
        rec = r["scoped_by_tier"].get("recovered_in_full_doc", {k: 0.0 for k in METRIC_KEYS})
        lines.append(
            f"| **`{r['strategy']}`** | {r['n_chunks']} | {r['n_questions_with_gt']} | {r['mean_top5_words']} | "
            f"{sc['hit@1']:.4f} | {sc['hit@3']:.4f} | **{sc['hit@5']:.4f}** | **{sc['mrr@5']:.4f}** | "
            f"{rec['hit@5']:.4f} / {rec['mrr@5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### 2.2. Nhánh Từ khóa Okapi BM25 (`k1=1.5, b=0.75`)",
            "",
            "| Chiến lược | Số chunks | Số câu có GT | Từ ngữ cảnh Top-5 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Nhóm câu khó `recovered` (Hit@5 / MRR@5) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in bm25_results:
        sc = r["scoped_evidence"]
        rec = r["scoped_by_tier"].get("recovered_in_full_doc", {k: 0.0 for k in METRIC_KEYS})
        lines.append(
            f"| **`{r['strategy']}`** | {r['n_chunks']} | {r['n_questions_with_gt']} | {r['mean_top5_words']} | "
            f"{sc['hit@1']:.4f} | {sc['hit@3']:.4f} | **{sc['hit@5']:.4f}** | **{sc['mrr@5']:.4f}** | "
            f"{rec['hit@5']:.4f} / {rec['mrr@5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Kết quả Truy xuất Toàn kho (`Global Retrieval` — 120 câu định vị ASIN & 91 câu trúng chunk)",
            "",
            "| Chiến lược | Phương pháp | Global ASIN Hit@1 | Global ASIN Hit@5 | Global ASIN MRR@5 | Global Chunk Hit@1 | Global Chunk Hit@5 | Global Chunk MRR@5 |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in vector_results:
        ga = r["global_asin"]
        ge = r["global_evidence"]
        lines.append(
            f"| `{r['strategy']}` | Vector (`BGE-small`) | {ga['hit@1']:.4f} | {ga['hit@5']:.4f} | {ga['mrr@5']:.4f} | "
            f"{ge['hit@1']:.4f} | {ge['hit@5']:.4f} | {ge['mrr@5']:.4f} |"
        )
    for r in bm25_results:
        ga = r["global_asin"]
        ge = r["global_evidence"]
        lines.append(
            f"| `{r['strategy']}` | BM25 | {ga['hit@1']:.4f} | {ga['hit@5']:.4f} | {ga['mrr@5']:.4f} | "
            f"{ge['hit@1']:.4f} | {ge['hit@5']:.4f} | {ge['mrr@5']:.4f} |"
        )

    base_vec = next((r for r in vector_results if r["strategy"] == "fixed"), None)
    if base_vec:
        lines.extend(
            [
                "",
                "## 4. Kiểm định Ý nghĩa Thống kê so với Baseline `fixed` (Nhánh Vector ASIN-scoped)",
                "",
                "| Chiến lược so với `fixed` | Số câu chung | $\\Delta$ Hit@5 | McNemar Hit@5 ($b / c$, $p$-value) | $\\Delta$ MRR@5 | Bootstrap 95% CI cho $\\Delta$ MRR@5 |",
                "| :--- | ---: | ---: | :--- | ---: | :--- |",
            ]
        )
        q_base = base_vec["scoped_by_qid"]
        for r in vector_results:
            if r["strategy"] == "fixed":
                continue
            q_new = r["scoped_by_qid"]
            common_qids = sorted(set(q_base.keys()) & set(q_new.keys()))
            h5_base = [q_base[q]["hit@5"] for q in common_qids]
            h5_new = [q_new[q]["hit@5"] for q in common_qids]
            mrr_base = [q_base[q]["mrr@5"] for q in common_qids]
            mrr_new = [q_new[q]["mrr@5"] for q in common_qids]
            mc = mcnemar_exact_pvalue(h5_base, h5_new)
            bt = bootstrap_diff_ci(mrr_base, mrr_new)
            dh5 = (sum(h5_new) - sum(h5_base)) / len(common_qids) if common_qids else 0.0
            lines.append(
                f"| `{r['strategy']}` vs. `fixed` | {len(common_qids)} | {dh5:+.4f} | "
                f"`b={mc['b_only_base']}, c={mc['c_only_new']}` ($p={mc['p_value']:.4f}$) | "
                f"{bt['diff_mean']:+.4f} | `[{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]` |"
            )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def chay_danh_gia_truy_xuat_6_chien_luoc(
    processed_dir: Path,
    eval_questions_path: Path,
    report_path: Path,
    strategies: Sequence[str] = ALL_STRATEGIES,
    embed_docs_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    embed_queries_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    top_k: int = 5,
    reuse_existing: bool = True,
) -> Dict:
    """Thực thi đánh giá BM25 và Dense Vector trên toàn bộ 6 chiến lược và xuất báo cáo."""
    _TEXT_EMB_MEMO.clear()
    cache_dir = processed_dir / "emb_cache"
    bm25_results: List[Dict] = []
    vector_results: List[Dict] = []

    if embed_docs_fn is None or embed_queries_fn is None:
        embedder = BGEFastEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=64)
        embed_docs_fn = embedder.embed_documents
        embed_queries_fn = embedder.embed_queries

    for st in strategies:
        chunks_path = processed_dir / f"chunks_dev_{st}.jsonl"
        children_path = (
            processed_dir / "chunks_dev_parent_child_children.jsonl"
            if st == "parent_child"
            else None
        )
        bm25_out = processed_dir / f"retrieval_bm25_dev_{st}.jsonl"
        vec_out = processed_dir / f"retrieval_vector_dev_{st}.jsonl"

        res_bm = danh_gia_bm25_chien_luoc(
            strategy=st,
            chunks_path=chunks_path,
            eval_questions_path=eval_questions_path,
            out_jsonl_path=bm25_out,
            children_path=children_path,
            top_k=top_k,
            reuse_existing=reuse_existing,
        )
        bm25_results.append(res_bm)

        res_vec = danh_gia_vector_chien_luoc(
            strategy=st,
            chunks_path=chunks_path,
            eval_questions_path=eval_questions_path,
            out_jsonl_path=vec_out,
            children_path=children_path,
            cache_dir=cache_dir,
            embed_docs_fn=embed_docs_fn,
            embed_queries_fn=embed_queries_fn,
            top_k=top_k,
            reuse_existing=reuse_existing,
        )
        vector_results.append(res_vec)
        print(
            f"[Vector {st}] chunks={res_vec['n_chunks']} "
            f"Scoped Hit@5={res_vec['scoped_evidence']['hit@5']:.4f} "
            f"MRR@5={res_vec['scoped_evidence']['mrr@5']:.4f} "
            f"Global ASIN Hit@5={res_vec['global_asin']['hit@5']:.4f}",
            flush=True,
        )

    ghi_bao_cao_so_sanh_truy_xuat(bm25_results, vector_results, report_path)
    return {
        "bm25": bm25_results,
        "vector": vector_results,
        "report_path": str(report_path),
    }
