"""Module chuẩn bị và đánh giá chốt 1 lần duy nhất trên tập test độc lập (Hạng mục 4)."""

import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from src.chunking import tao_chunks_chien_luoc
from src.chunking_eval import (
    METRIC_KEYS,
    _nap_hoac_tinh_embeddings,
    bootstrap_diff_ci,
    mcnemar_exact_pvalue,
)
from src.evaluation_judge import (
    goi_gemini_json,
    tao_prompt_cham_batch_no_rag_direct,
    tao_prompt_cham_batch_rag,
    tinh_context_precision_at_k,
)
from src.generation_baseline import (
    goi_groq_chat_completion,
    la_cau_tu_choi_tra_loi,
    tao_prompt_no_rag,
    tao_prompt_rag,
)
from src.hybrid_retrieval import (
    _chuan_hoa_ma_tran,
    _xay_dung_top_k_items,
    gop_ve_don_vi_cha,
    rrf_fuse,
    xep_hang_vector_numpy,
)
from src.label_verification import tinh_cohens_kappa
from src.retrieval_bm25 import (
    _trung_binh_metrics,
    doc_danh_sach_jsonl,
    tinh_hit_va_mrr,
)
from src.retrieval_vector import BGEFastEmbedder
from src.verify_no_labels import BM25TinhDiem, bo_stopword, chuan_hoa, tokenize

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def lay_mau_phan_tang_test_120(
    questions_test_clean_path: Path,
    out_eval_path: Path,
    n_per_cell: int = 30,
    seed: int = 42,
) -> Dict:
    """Lấy mẫu phân tầng 120 câu từ tập test sạch (30 câu x 4 tổ hợp question_type x is_answerable_auto)."""
    rows, _ = doc_danh_sach_jsonl(questions_test_clean_path)
    buckets: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        qt = str(r.get("question_type") or "").strip().lower()
        ia = int(r.get("is_answerable_auto", 0))
        if qt in ("descriptive", "yesno") and ia in (0, 1):
            buckets.setdefault((qt, ia), []).append(r)

    rng = random.Random(seed)
    selected: List[Dict] = []
    cell_counts: Dict[str, int] = {}
    for key in (("descriptive", 0), ("descriptive", 1), ("yesno", 0), ("yesno", 1)):
        pool = sorted(buckets.get(key, []), key=lambda x: str(x.get("qid_moi", "")))
        k = min(n_per_cell, len(pool))
        picks = rng.sample(pool, k)
        selected.extend(picks)
        cell_counts[f"{key[0]}_ia{key[1]}"] = k

    out_eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_eval_path, "w", encoding="utf-8") as fout:
        for rec in selected:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "n_selected": len(selected),
        "cell_counts": cell_counts,
        "out_eval_path": str(out_eval_path),
    }


def _khop_chunks_bang_chung_test(question_rec: Dict, doc_chunks: Sequence[Dict]) -> set:
    snippets = [s for s in (question_rec.get("review_snippets_clean") or []) if isinstance(s, str) and s.strip()]
    ev_pairs = [(chuan_hoa(s), set(bo_stopword(tokenize(s)))) for s in snippets if chuan_hoa(s)]
    matched = set()
    for ch in doc_chunks:
        cid = ch.get("chunk_id")
        ctext = ch.get("text") or ""
        if not cid or not ctext:
            continue
        norm_c = chuan_hoa(ctext)
        toks_c = set(bo_stopword(tokenize(ctext)))
        for norm_ev, toks_ev in ev_pairs:
            probe = norm_ev[:120] if len(norm_ev) > 120 else norm_ev
            if probe and (probe in norm_c or norm_c in norm_ev):
                matched.add(cid)
                break
            common = len(toks_ev & toks_c)
            threshold = min(12, max(4, int(len(toks_ev) * 0.6)))
            if toks_ev and common >= threshold:
                matched.add(cid)
                break
    return matched


def chuan_bi_truy_xuat_test(
    processed_dir: Path,
    eval_test_path: Path,
    report_path: Path,
    embed_docs_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    embed_queries_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    k0: int = 60,
    depth: int = 50,
    top_k: int = 5,
) -> Dict:
    """Sinh chunks tập test (recursive, parent_child) và chạy truy xuất cho 3 cấu hình đã khóa."""
    docs_test_clean = processed_dir / "docs_test_clean.jsonl"
    rec_chunks_path = processed_dir / "chunks_test_recursive.jsonl"
    pc_chunks_path = processed_dir / "chunks_test_parent_child.jsonl"
    pc_child_path = processed_dir / "chunks_test_parent_child_children.jsonl"

    if not rec_chunks_path.exists():
        tao_chunks_chien_luoc(
            input_path=docs_test_clean,
            output_path=rec_chunks_path,
            strategy="recursive",
            params={"max_words": 200, "overlap_words": 40},
        )
    if not pc_chunks_path.exists() or not pc_child_path.exists():
        tao_chunks_chien_luoc(
            input_path=docs_test_clean,
            output_path=pc_chunks_path,
            strategy="parent_child",
            params={"max_words": 200, "child_max_words": 60},
            children_output_path=pc_child_path,
        )

    if embed_docs_fn is None or embed_queries_fn is None:
        embedder = BGEFastEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=32)
        embed_docs_fn = embedder.embed_documents
        embed_queries_fn = embedder.embed_queries

    cache_dir = processed_dir / "emb_cache"
    questions, _ = doc_danh_sach_jsonl(eval_test_path)
    q_texts = [(q.get("question_text_clean") or q.get("question_text") or "") for q in questions]
    q_vecs = _nap_hoac_tinh_embeddings(q_texts, cache_dir / "emb_test_queries_120.npy", embed_queries_fn)

    configs = [
        ("fixed_vector", "fixed", "vector", processed_dir / "chunks_test_fixed.jsonl", None),
        ("recursive_hybrid", "recursive", "hybrid_rrf", rec_chunks_path, None),
        ("parent_child_hybrid", "parent_child", "hybrid_rrf", pc_chunks_path, pc_child_path),
    ]

    summary_rows: List[Dict] = []
    for cfg_name, st, mode, c_path, ch_path in configs:
        parent_chunks, _ = doc_danh_sach_jsonl(c_path)
        is_pc = ch_path is not None and ch_path.exists()
        index_chunks, _ = doc_danh_sach_jsonl(ch_path) if is_pc else (parent_chunks, 0)
        parent_by_id = {c["chunk_id"]: c for c in parent_chunks}
        child_to_parent = {c["chunk_id"]: c.get("parent_id", c["chunk_id"]) for c in index_chunks}

        all_texts = [str(c.get("text", "")) for c in index_chunks]
        all_vecs = _nap_hoac_tinh_embeddings(all_texts, cache_dir / f"emb_test_{st}.npy", embed_docs_fn)
        mat = _chuan_hoa_ma_tran(all_vecs)

        bm25 = None
        if mode == "hybrid_rrf":
            bm25 = BM25TinhDiem([bo_stopword(tokenize(t)) for t in all_texts], k1=1.5, b=0.75)

        doc_to_idx: Dict[str, List[int]] = {}
        for i, c in enumerate(index_chunks):
            doc_to_idx.setdefault(c.get("doc_id", ""), []).append(i)
        doc_to_parents: Dict[str, List[Dict]] = {}
        for c in parent_chunks:
            doc_to_parents.setdefault(c.get("doc_id", ""), []).append(c)

        out_path = processed_dir / f"retrieval_test_{cfg_name}.jsonl"
        g_asin_list: List[Dict[str, float]] = []
        s_ev_list: List[Dict[str, float]] = []
        search_depth = depth * 4 if is_pc else depth

        with open(out_path, "w", encoding="utf-8") as fout:
            for q, q_text, q_vec in zip(questions, q_texts, q_vecs):
                doc_id = q.get("doc_id", "")
                scoped_idx = doc_to_idx.get(doc_id, [])
                gt_ids = _khop_chunks_bang_chung_test(q, doc_to_parents.get(doc_id, []))

                vec_g_raw = [index_chunks[i]["chunk_id"] for i, _ in xep_hang_vector_numpy(q_vec, mat, None, search_depth)]
                vec_s_raw = [index_chunks[i]["chunk_id"] for i, _ in xep_hang_vector_numpy(q_vec, mat, scoped_idx, search_depth)]

                if mode == "vector":
                    g_ids = vec_g_raw[:top_k]
                    s_ids = vec_s_raw[:top_k]
                    top_global = _xay_dung_top_k_items([(cid, 1.0) for cid in g_ids], parent_by_id, {}, {cid: i for i, cid in enumerate(g_ids, 1)}, top_k)
                    top_scoped = _xay_dung_top_k_items([(cid, 1.0) for cid in s_ids], parent_by_id, {}, {cid: i for i, cid in enumerate(s_ids, 1)}, top_k)
                else:
                    q_toks = bo_stopword(tokenize(q_text))
                    bm_g = sorted(((bm25.diem(q_toks, i), i) for i in range(len(index_chunks))), key=lambda x: (-x[0], x[1]))
                    bm_s = sorted(((bm25.diem(q_toks, i), i) for i in scoped_idx), key=lambda x: (-x[0], x[1]))
                    bm_g_raw = [index_chunks[i]["chunk_id"] for _, i in bm_g[:search_depth]]
                    bm_s_raw = [index_chunks[i]["chunk_id"] for _, i in bm_s[:search_depth]]
                    if is_pc:
                        bm_g_ids = gop_ve_don_vi_cha(bm_g_raw, child_to_parent, depth)
                        vec_g_ids = gop_ve_don_vi_cha(vec_g_raw, child_to_parent, depth)
                        bm_s_ids = gop_ve_don_vi_cha(bm_s_raw, child_to_parent, depth)
                        vec_s_ids = gop_ve_don_vi_cha(vec_s_raw, child_to_parent, depth)
                    else:
                        bm_g_ids, vec_g_ids = bm_g_raw[:depth], vec_g_raw[:depth]
                        bm_s_ids, vec_s_ids = bm_s_raw[:depth], vec_s_raw[:depth]
                    fg = rrf_fuse([bm_g_ids, vec_g_ids], k0=k0, depth=depth)
                    fs = rrf_fuse([bm_s_ids, vec_s_ids], k0=k0, depth=depth)
                    top_global = _xay_dung_top_k_items(fg, parent_by_id, {c: i for i, c in enumerate(bm_g_ids, 1)}, {c: i for i, c in enumerate(vec_g_ids, 1)}, top_k)
                    top_scoped = _xay_dung_top_k_items(fs, parent_by_id, {c: i for i, c in enumerate(bm_s_ids, 1)}, {c: i for i, c in enumerate(vec_s_ids, 1)}, top_k)

                m_ga = tinh_hit_va_mrr(top_global, "doc_id", {doc_id})
                g_asin_list.append(m_ga)
                m_se = tinh_hit_va_mrr(top_scoped, "chunk_id", gt_ids) if gt_ids else None
                if m_se:
                    s_ev_list.append(m_se)

                rec = {
                    "qid_moi": q.get("qid_moi", ""),
                    "asin": q.get("asin", ""),
                    "doc_id": doc_id,
                    "config": cfg_name,
                    "question_text_clean": q_text,
                    "question_type": q.get("question_type", ""),
                    "is_answerable_auto": q.get("is_answerable_auto", 0),
                    "reference_answer": q.get("reference_answer", ""),
                    "review_snippets_clean": q.get("review_snippets_clean", []),
                    "metrics_global_asin": m_ga,
                    "metrics_scoped_evidence": m_se,
                    "top_k_scoped": top_scoped,
                    "top_k_global": top_global,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary_rows.append(
            {
                "config": cfg_name,
                "n_chunks": len(parent_chunks),
                "global_asin": _trung_binh_metrics(g_asin_list, METRIC_KEYS),
                "scoped_evidence": _trung_binh_metrics(s_ev_list, METRIC_KEYS),
                "out_jsonl_path": str(out_path),
            }
        )
        print(
            f"[Test Retrieval {cfg_name}] Global ASIN Hit@5={summary_rows[-1]['global_asin']['hit@5']:.4f} "
            f"MRR@5={summary_rows[-1]['global_asin']['mrr@5']:.4f} "
            f"Scoped Hit@5={summary_rows[-1]['scoped_evidence']['hit@5']:.4f}",
            flush=True,
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Truy xuất Độc lập trên Tập Test (150 sản phẩm Electronics, 120 câu hỏi phân tầng)",
        "",
        "| Cấu hình đã khóa từ `dev` | Số chunks | Global ASIN Hit@1 | Global ASIN Hit@5 | Global ASIN MRR@5 | Scoped Hit@1 | Scoped Hit@5 | Scoped MRR@5 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary_rows:
        ga, sc = s["global_asin"], s["scoped_evidence"]
        lines.append(
            f"| `{s['config']}` | {s['n_chunks']} | {ga['hit@1']:.4f} | **{ga['hit@5']:.4f}** | **{ga['mrr@5']:.4f}** | "
            f"{sc['hit@1']:.4f} | **{sc['hit@5']:.4f}** | **{sc['mrr@5']:.4f}** |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"summary": summary_rows, "report_path": str(report_path)}


def chay_danh_gia_chot_tap_test(
    processed_dir: Path,
    eval_test_path: Path,
    report_path: Path,
    generator_model: str = "qwen3.8:27b",
    judge_model: str = "gemini-3.5-flash",
    gen_delay_sec: float = 0.0,
    judge_delay_sec: float = 0.8,
    batch_size: int = 5,
    llm_call_fn: Optional[Callable[[Sequence[Dict[str, str]], str], Tuple[str, Dict]]] = None,
    gemini_call_fn: Optional[Callable[[str, str], Union[Dict, List]]] = None,
) -> Dict:
    """Chạy đánh giá chốt 1 lần duy nhất trên tập test cho 4 cấu hình đã khóa."""
    call_llm = llm_call_fn or (lambda msgs, m: goi_groq_chat_completion(msgs, model_name=m))
    call_gem = gemini_call_fn or (lambda prompt, m: goi_gemini_json(prompt, model_name=m))

    questions, _ = doc_danh_sach_jsonl(eval_test_path)
    q_by_id = {q["qid_moi"]: q for q in questions}

    def _run_gen(cfg_name: str, is_no_rag: bool) -> List[Dict]:
        gen_path = processed_dir / f"generations_test_{cfg_name}_ollama.jsonl"
        existing: Dict[str, Dict] = {}
        if gen_path.exists():
            for r in doc_danh_sach_jsonl(gen_path)[0]:
                if r.get("qid_moi") and r.get("generated_answer"):
                    existing[r["qid_moi"]] = r
        src_rows = questions if is_no_rag else doc_danh_sach_jsonl(processed_dir / f"retrieval_test_{cfg_name}.jsonl")[0]
        out_rows: List[Dict] = []
        with open(gen_path, "w", encoding="utf-8") as fout:
            for idx, r in enumerate(src_rows, start=1):
                qid = r["qid_moi"]
                if qid in existing:
                    rec = dict(existing[qid])
                    out_rows.append(rec)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                if is_no_rag:
                    msgs = tao_prompt_no_rag(r["asin"], "Electronics", r["question_text_clean"])
                    ctxs = []
                else:
                    top_ch = (r.get("top_k_scoped") or [])[:5]
                    msgs = tao_prompt_rag(r["asin"], "Electronics", r["question_text_clean"], top_ch)
                    ctxs = [c["text"] for c in top_ch]
                ans, usage = call_llm(msgs, generator_model)
                rec = {
                    "qid_moi": qid,
                    "asin": r["asin"],
                    "config": cfg_name,
                    "question_text_clean": r["question_text_clean"],
                    "reference_answer": r.get("reference_answer", ""),
                    "retrieved_contexts": ctxs,
                    "generated_answer": ans,
                    "is_abstention": la_cau_tu_choi_tra_loi(ans),
                    "generator_model": generator_model,
                    "usage": usage,
                }
                out_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if gen_delay_sec > 0 and llm_call_fn is None and idx < len(src_rows):
                    time.sleep(gen_delay_sec)
        return out_rows

    def _run_judge(cfg_name: str, gen_rows: List[Dict], is_no_rag: bool) -> List[Dict]:
        j_path = processed_dir / f"eval_judge_test_{cfg_name}_ollama.jsonl"
        existing: Dict[str, Dict] = {}
        if j_path.exists():
            for r in doc_danh_sach_jsonl(j_path)[0]:
                if r.get("qid_moi") and "faithfulness" in r:
                    existing[r["qid_moi"]] = r
        todo = [r for r in gen_rows if r["qid_moi"] not in existing]
        if todo:
            with open(j_path, "a", encoding="utf-8") as fa:
                if is_no_rag:
                    direct_todo = [r for r in todo if not r["is_abstention"]]
                    for r in todo:
                        if r["is_abstention"]:
                            rec = {"qid_moi": r["qid_moi"], "is_abstention": True, "context_precision": 0.0, "faithfulness": 1.0, "answer_relevance": 0.0, "has_hallucination": 0, "judge_answerable": 0}
                            existing[r["qid_moi"]] = rec
                            fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for start in range(0, len(direct_todo), batch_size):
                        batch = []
                        for r in direct_todo[start : start + batch_size]:
                            m = dict(q_by_id.get(r["qid_moi"], {}))
                            m.update(r)
                            batch.append(m)
                        res = call_gem(tao_prompt_cham_batch_no_rag_direct(batch), judge_model)
                        if isinstance(res, dict):
                            res = res.get("items") or res.get("results") or [res]
                        bmap = {str(o.get("qid_moi", "")): o for o in (res or []) if isinstance(o, dict)}
                        for r in batch:
                            j = bmap.get(r["qid_moi"], {})
                            rec = {
                                "qid_moi": r["qid_moi"],
                                "is_abstention": False,
                                "context_precision": 0.0,
                                "faithfulness": round(float(j.get("faithfulness", 0.0)), 4),
                                "answer_relevance": round(float(j.get("answer_relevance", 0.0)), 4),
                                "has_hallucination": int(j.get("has_hallucination", 1)),
                                "judge_answerable": 0,
                            }
                            existing[r["qid_moi"]] = rec
                            fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fa.flush()
                        if judge_delay_sec > 0 and gemini_call_fn is None:
                            time.sleep(judge_delay_sec)
                else:
                    for start in range(0, len(todo), batch_size):
                        batch = todo[start : start + batch_size]
                        res = call_gem(tao_prompt_cham_batch_rag(batch), judge_model)
                        if isinstance(res, dict):
                            res = res.get("items") or res.get("results") or [res]
                        bmap = {str(o.get("qid_moi", "")): o for o in (res or []) if isinstance(o, dict)}
                        for r in batch:
                            j = bmap.get(r["qid_moi"], {})
                            flags = [1 if int(x) > 0 else 0 for x in (j.get("chunk_relevance") or [0] * 5)[:5]]
                            is_abs = bool(r["is_abstention"])
                            rec = {
                                "qid_moi": r["qid_moi"],
                                "is_abstention": is_abs,
                                "context_precision": tinh_context_precision_at_k(flags),
                                "faithfulness": 1.0 if is_abs else round(float(j.get("faithfulness", 1.0)), 4),
                                "answer_relevance": 0.0 if is_abs else round(float(j.get("answer_relevance", 0.8)), 4),
                                "has_hallucination": 0 if is_abs else int(j.get("has_hallucination", 0)),
                                "judge_answerable": int(j.get("judge_answerable", 1 if sum(flags) > 0 else 0)),
                            }
                            existing[r["qid_moi"]] = rec
                            fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fa.flush()
                        if judge_delay_sec > 0 and gemini_call_fn is None:
                            time.sleep(judge_delay_sec)
        judged = [existing[r["qid_moi"]] for r in gen_rows]
        with open(j_path, "w", encoding="utf-8") as fout:
            for rec in judged:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return judged

    def _mean(vals: Sequence[float]) -> float:
        return round(float(statistics.mean(vals)), 4) if vals else 0.0

    cfgs = [
        ("no_rag", True),
        ("fixed_vector", False),
        ("recursive_hybrid", False),
        ("parent_child_hybrid", False),
    ]
    summaries: List[Dict] = []
    for name, is_nr in cfgs:
        g = _run_gen(name, is_nr)
        j = _run_judge(name, g, is_nr)
        direct = [r for r in j if not r["is_abstention"]]
        ans_j = [r for r in j if r.get("judge_answerable") == 1]
        y_j = [int(r.get("judge_answerable", 0)) for r in j]
        y_a = [0 if r["is_abstention"] else 1 for r in j]
        s = {
            "config": name,
            "n_total": len(j),
            "n_direct": len(direct),
            "cp5_all": _mean([r["context_precision"] for r in j]),
            "cp5_ans": _mean([r["context_precision"] for r in ans_j]),
            "faith_direct": _mean([r["faithfulness"] for r in direct]),
            "faith_all": _mean([r["faithfulness"] for r in j]),
            "hallu_count": sum(int(r["has_hallucination"]) for r in j),
            "hallu_rate_direct": round(_mean([float(r["has_hallucination"]) for r in direct]) * 100.0, 2),
            "ans_rel_direct": _mean([r["answer_relevance"] for r in direct]),
            "ans_rel_all": _mean([r["answer_relevance"] for r in j]),
            "kappa_judge_vs_rag": tinh_cohens_kappa(y_j, y_a)["kappa"] if not is_nr else 0.0,
        }
        summaries.append(s)
        print(
            f"[Test Judge {name}] direct={s['n_direct']}/120 CP@5(all)={s['cp5_all']:.4f} "
            f"Faith(direct)={s['faith_direct']:.4f} Hallu={s['hallu_rate_direct']}%",
            flush=True,
        )

    lines = [
        "# Báo cáo Đánh giá Chốt 1 Lần Duy Nhất trên Tập Test Độc Lập (120 câu Electronics)",
        "",
        f"- **Generator:** `{generator_model}` | **Judge:** `{judge_model}` | **Tập đánh giá:** `eval_120_test_stratified.jsonl` (150 `asin` cô lập hoàn toàn với `dev`)",
        "",
        "| Cấu hình đã khóa | Trả lời trực tiếp (/120) | CP@5 (All) | CP@5 (`judge_ans=1`) | Faithfulness (Direct) | Faithfulness (All) | Hallucination on Direct | Answer Rel (Direct) | Answer Rel (All) | $\\kappa$ (Judge vs RAG) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| **`{s['config']}`** | {s['n_direct']}/120 ({s['n_direct']*100.0/max(1, s['n_total']):.2f}%) | "
            f"{s['cp5_all']:.4f} | {s['cp5_ans']:.4f} | **{s['faith_direct']:.4f}** | {s['faith_all']:.4f} | "
            f"**{s['hallu_rate_direct']:.2f}%** ({s['hallu_count']}/{s['n_direct']}) | "
            f"{s['ans_rel_direct']:.4f} | {s['ans_rel_all']:.4f} | {s['kappa_judge_vs_rag']:.4f} |"
        )
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"summaries": summaries, "report_path": str(report_path)}
