"""Module truy xuất cơ sở bằng Dense Vector (BAAI/bge-small-en-v1.5 + ChromaDB)."""

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from src.retrieval_bm25 import (
    _trung_binh_metrics,
    doc_danh_sach_jsonl,
    tinh_hit_va_mrr,
    xac_dinh_ground_truth_chunks,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent


class BGEFastEmbedder:
    """Bộ nhúng văn bản dùng FastEmbed (ONNX Runtime) cho BAAI/bge-small-en-v1.5."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]
        sorted_vecs = [v.tolist() for v in self.model.passage_embed(sorted_texts, batch_size=self.batch_size)]
        out: List[List[float]] = [[] for _ in range(len(texts))]
        for orig_idx, vec in zip(order, sorted_vecs):
            out[orig_idx] = vec
        return out

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        if not queries:
            return []
        vecs = list(self.model.query_embed(list(queries), batch_size=self.batch_size))
        return [v.tolist() for v in vecs]


def tao_hoac_nap_chroma_collection(
    chunks: Sequence[Dict],
    embed_docs_fn: Callable[[Sequence[str]], List[List[float]]],
    collection_name: str = "amazonqa_electronics_dev_fixed",
    persist_dir: Optional[Path] = None,
    batch_size: int = 256,
    force_rebuild: bool = True,
):
    """Khởi tạo ChromaDB collection (cosine) và nạp danh sách chunks kèm vector."""
    import chromadb

    if persist_dir is not None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
    else:
        client = chromadb.EphemeralClient()

    if force_rebuild:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == len(chunks) and len(chunks) > 0 and not force_rebuild:
        return collection

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        ids = [str(c["chunk_id"]) for c in batch]
        docs = [str(c.get("text", "")) for c in batch]
        metas = [
            {
                "doc_id": str(c.get("doc_id", "")),
                "asin": str(c.get("asin", "")),
                "category": str(c.get("category", "")),
                "chunk_index": int(c.get("chunk_index", 0)),
                "n_words": int(c.get("n_words", 0)),
            }
            for c in batch
        ]
        embeddings = embed_docs_fn(docs)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
        )

    return collection


def truy_xuat_chroma_top_k(
    collection,
    query_vec: Sequence[float],
    top_k: int = 5,
    where_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Truy vấn top_k đoạn văn gần nhất trên ChromaDB theo Cosine Similarity."""
    kwargs = {
        "query_embeddings": [list(query_vec)],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    res = collection.query(**kwargs)
    ids_list = (res.get("ids") or [[]])[0]
    docs_list = (res.get("documents") or [[]])[0]
    metas_list = (res.get("metadatas") or [[]])[0]
    dists_list = (res.get("distances") or [[]])[0]

    out: List[Dict] = []
    for rank_idx, (cid, doc_text, meta, dist) in enumerate(
        zip(ids_list, docs_list, metas_list, dists_list), start=1
    ):
        meta = meta or {}
        sim = 1.0 - float(dist)
        out.append(
            {
                "rank": rank_idx,
                "score": round(sim, 4),
                "chunk_id": str(cid),
                "doc_id": str(meta.get("doc_id", "")),
                "asin": str(meta.get("asin", "")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "n_words": int(meta.get("n_words", 0)),
                "text": str(doc_text or ""),
            }
        )
    return out


def chay_danh_gia_vector_baseline(
    chunks_path: Path,
    eval_questions_path: Path,
    out_jsonl_path: Path,
    report_path: Path,
    persist_dir: Optional[Path] = None,
    model_name: str = "BAAI/bge-small-en-v1.5",
    top_k: int = 5,
    embed_docs_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    embed_queries_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    bm25_jsonl_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Dict:
    """Chạy truy xuất Dense Vector (BAAI/bge-small-en-v1.5 + ChromaDB) cho 120 câu hỏi."""
    t0 = time.time()
    chunks, n_corrupt_chunks = doc_danh_sach_jsonl(chunks_path)
    questions, n_corrupt_q = doc_danh_sach_jsonl(eval_questions_path)

    if embed_docs_fn is None or embed_queries_fn is None:
        embedder = BGEFastEmbedder(model_name=model_name)
        embed_docs_fn = embedder.embed_documents
        embed_queries_fn = embedder.embed_queries

    collection = tao_hoac_nap_chroma_collection(
        chunks=chunks,
        embed_docs_fn=embed_docs_fn,
        collection_name="amazonqa_electronics_dev_fixed",
        persist_dir=persist_dir,
        force_rebuild=force_rebuild,
    )
    t_index = round(time.time() - t0, 2)

    doc_to_chunks: Dict[str, List[Dict]] = {}
    for c in chunks:
        doc_to_chunks.setdefault(c.get("doc_id", ""), []).append(c)

    q_texts = [
        (q.get("question_text_clean") or q.get("question_text") or "")
        for q in questions
    ]
    q_vecs = embed_queries_fn(q_texts) if q_texts else []

    metric_keys = ("hit@1", "hit@3", "hit@5", "mrr@5")
    global_asin_metrics_all: List[Dict[str, float]] = []
    global_ev_metrics_ans: List[Dict[str, float]] = []
    scoped_ev_metrics_ans: List[Dict[str, float]] = []
    scoped_by_tier: Dict[str, List[Dict[str, float]]] = {}

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n_questions_with_gt = 0

    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for q, q_text, q_vec in zip(questions, q_texts, q_vecs):
            qid = q.get("qid_moi", "")
            doc_id = q.get("doc_id", "")
            asin = q.get("asin", "")
            nhan_3 = q.get("nhan_chot_3_muc", "")
            nhan_ct = q.get("nhan_chot_chi_tiet", "")

            same_doc_chunks = doc_to_chunks.get(doc_id, [])
            gt_chunk_ids = xac_dinh_ground_truth_chunks(q, same_doc_chunks)

            top_global = truy_xuat_chroma_top_k(
                collection=collection,
                query_vec=q_vec,
                top_k=top_k,
                where_filter=None,
            )
            n_scoped = min(top_k, max(1, len(same_doc_chunks)))
            top_scoped = (
                truy_xuat_chroma_top_k(
                    collection=collection,
                    query_vec=q_vec,
                    top_k=n_scoped,
                    where_filter={"doc_id": doc_id},
                )
                if same_doc_chunks
                else []
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

    t_total = round(time.time() - t0, 2)
    avg_global_asin = _trung_binh_metrics(global_asin_metrics_all, metric_keys)
    avg_global_ev = _trung_binh_metrics(global_ev_metrics_ans, metric_keys)
    avg_scoped_ev = _trung_binh_metrics(scoped_ev_metrics_ans, metric_keys)
    avg_by_tier = {
        tier: _trung_binh_metrics(rows, metric_keys)
        for tier, rows in sorted(scoped_by_tier.items())
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Truy xuất Cơ sở bằng Dense Vector (`BAAI/bge-small-en-v1.5` + `ChromaDB`)",
        "",
        "## 1. Cấu hình thực nghiệm",
        f"- Mô hình Embedding: **`{model_name}`** (384 chiều, chuẩn hóa Cosine, thực thi qua ONNX Runtime / FastEmbed)",
        "- Cơ sở dữ liệu Vector: **`ChromaDB`** (`hnsw:space = cosine`)",
        f"- Tập đoạn văn cơ sở: `{chunks_path.name}` (**{len(chunks)} chunks** từ 150 sản phẩm `Electronics`, `fixed_size` 200 từ, overlap 40 từ)",
        f"- Tập câu hỏi đánh giá: `{eval_questions_path.name}` (**{len(questions)} câu hỏi**)",
        f"- Thời gian nhúng & lập chỉ mục {len(chunks)} chunks: **{t_index} giây** (Tổng thời gian chạy: **{t_total} giây**)",
        "",
        "## 2. Kết quả Truy xuất Dense Vector (Hit@1, Hit@3, Hit@5, MRR@5)",
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

    if bm25_jsonl_path is not None and bm25_jsonl_path.exists():
        bm25_rows, _ = doc_danh_sach_jsonl(bm25_jsonl_path)
        bm_g_asin = [r["metrics_global_asin"] for r in bm25_rows if r.get("metrics_global_asin")]
        bm_g_ev = [r["metrics_global_evidence"] for r in bm25_rows if r.get("metrics_global_evidence")]
        bm_s_ev = [r["metrics_scoped_evidence"] for r in bm25_rows if r.get("metrics_scoped_evidence")]
        bm_rec = [
            r["metrics_scoped_evidence"]
            for r in bm25_rows
            if r.get("nhan_chot_chi_tiet") == "recovered_in_full_doc" and r.get("metrics_scoped_evidence")
        ]
        if bm_g_asin and bm_s_ev:
            a_bm_ga = _trung_binh_metrics(bm_g_asin, metric_keys)
            a_bm_ge = _trung_binh_metrics(bm_g_ev, metric_keys)
            a_bm_se = _trung_binh_metrics(bm_s_ev, metric_keys)
            a_bm_rec = _trung_binh_metrics(bm_rec, metric_keys)
            vec_rec = avg_by_tier.get("recovered_in_full_doc", {k: 0.0 for k in metric_keys})
            lines.extend(
                [
                    "## 4. Đối sánh trực tiếp: BM25 (Lexical) vs. BAAI/bge-small-en-v1.5 (Dense Vector)",
                    "",
                    "| Kịch bản đánh giá | Phương pháp | Hit@1 | Hit@3 | Hit@5 | MRR@5 |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: |",
                    f"| **Toàn kho (`Global` — Định vị đúng `ASIN`, 120 câu)** | BM25 (Từ khóa) | {a_bm_ga['hit@1']:.4f} | {a_bm_ga['hit@3']:.4f} | {a_bm_ga['hit@5']:.4f} | {a_bm_ga['mrr@5']:.4f} |",
                    f"| **Toàn kho (`Global` — Định vị đúng `ASIN`, 120 câu)** | **BGE-small-en-v1.5 (Vector)** | **{avg_global_asin['hit@1']:.4f}** | **{avg_global_asin['hit@3']:.4f}** | **{avg_global_asin['hit@5']:.4f}** | **{avg_global_asin['mrr@5']:.4f}** |",
                    f"| **Toàn kho (`Global` — Trúng `chunk` bằng chứng, 91 câu)** | BM25 (Từ khóa) | {a_bm_ge['hit@1']:.4f} | {a_bm_ge['hit@3']:.4f} | {a_bm_ge['hit@5']:.4f} | {a_bm_ge['mrr@5']:.4f} |",
                    f"| **Toàn kho (`Global` — Trúng `chunk` bằng chứng, 91 câu)** | **BGE-small-en-v1.5 (Vector)** | **{avg_global_ev['hit@1']:.4f}** | **{avg_global_ev['hit@3']:.4f}** | **{avg_global_ev['hit@5']:.4f}** | **{avg_global_ev['mrr@5']:.4f}** |",
                    f"| **Trong sản phẩm (`ASIN-scoped` — 91 câu có bằng chứng)** | BM25 (Từ khóa) | {a_bm_se['hit@1']:.4f} | {a_bm_se['hit@3']:.4f} | {a_bm_se['hit@5']:.4f} | {a_bm_se['mrr@5']:.4f} |",
                    f"| **Trong sản phẩm (`ASIN-scoped` — 91 câu có bằng chứng)** | **BGE-small-en-v1.5 (Vector)** | **{avg_scoped_ev['hit@1']:.4f}** | **{avg_scoped_ev['hit@3']:.4f}** | **{avg_scoped_ev['hit@5']:.4f}** | **{avg_scoped_ev['mrr@5']:.4f}** |",
                    f"| **Nhóm câu khó (`recovered_in_full_doc`, 19 câu)** | BM25 (Từ khóa) | {a_bm_rec['hit@1']:.4f} | {a_bm_rec['hit@3']:.4f} | {a_bm_rec['hit@5']:.4f} | {a_bm_rec['mrr@5']:.4f} |",
                    f"| **Nhóm câu khó (`recovered_in_full_doc`, 19 câu)** | **BGE-small-en-v1.5 (Vector)** | **{vec_rec['hit@1']:.4f}** | **{vec_rec['hit@3']:.4f}** | **{vec_rec['hit@5']:.4f}** | **{vec_rec['mrr@5']:.4f}** |",
                    "",
                ]
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "n_questions_with_gt": n_questions_with_gt,
        "n_corrupt_chunks": n_corrupt_chunks,
        "n_corrupt_questions": n_corrupt_q,
        "index_time_sec": t_index,
        "total_time_sec": t_total,
        "global_asin": avg_global_asin,
        "global_evidence": avg_global_ev,
        "scoped_evidence": avg_scoped_ev,
        "scoped_by_tier": avg_by_tier,
        "out_jsonl_path": str(out_jsonl_path),
        "report_path": str(report_path),
    }
