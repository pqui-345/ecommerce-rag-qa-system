"""Sinh câu trả lời (Groq Qwen) và chấm điểm RAGAS (Gemini Judge) đối sánh các chiến lược phân đoạn (RQ1 & RQ2)."""

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from src.chunking_eval import mcnemar_exact_pvalue
from src.evaluation_judge import (
    goi_gemini_json,
    tao_prompt_cham_batch_rag,
    tinh_context_precision_at_k,
)
from src.generation_baseline import (
    goi_groq_chat_completion,
    la_cau_tu_choi_tra_loi,
    tao_prompt_rag,
)
from src.label_verification import tinh_cohens_kappa
from src.retrieval_bm25 import doc_danh_sach_jsonl

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def sinh_cau_tra_loi_chien_luoc(
    strategy: str,
    retrieval_jsonl_path: Path,
    out_jsonl_path: Path,
    model_name: str = "qwen/qwen3.8-27b",
    top_k: int = 5,
    delay_sec: float = 1.0,
    llm_call_fn: Optional[Callable[[Sequence[Dict[str, str]], str], Tuple[str, Dict]]] = None,
) -> List[Dict]:
    """Sinh câu trả lời RAG cho một chiến lược phân đoạn, hỗ trợ resume từng dòng."""
    ret_rows, _ = doc_danh_sach_jsonl(retrieval_jsonl_path)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    existing_by_qid: Dict[str, Dict] = {}
    if out_jsonl_path.exists():
        old_rows, _ = doc_danh_sach_jsonl(out_jsonl_path)
        for r in old_rows:
            qid = r.get("qid_moi")
            if qid and r.get("generated_answer"):
                existing_by_qid[qid] = r

    call_fn = llm_call_fn or (lambda msgs, m: goi_groq_chat_completion(msgs, model_name=m))
    mode_name = f"naive_rag_{strategy}"

    results: List[Dict] = []
    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(ret_rows, start=1):
            qid = row.get("qid_moi", "")
            if qid in existing_by_qid:
                rec = dict(existing_by_qid[qid])
                rec["is_abstention"] = la_cau_tu_choi_tra_loi(rec.get("generated_answer", ""))
                results.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            asin = row.get("asin", "")
            doc_id = row.get("doc_id", "")
            q_text = row.get("question_text_clean") or ""
            top_chunks = (row.get("top_k_scoped") or [])[:top_k]

            msgs = tao_prompt_rag(
                asin=asin,
                category="Electronics",
                question_text=q_text,
                retrieved_chunks=top_chunks,
            )
            ans_text, usage_meta = call_fn(msgs, model_name)
            is_abstain = la_cau_tu_choi_tra_loi(ans_text)

            rec = {
                "qid_moi": qid,
                "asin": asin,
                "doc_id": doc_id,
                "mode": mode_name,
                "strategy": strategy,
                "generator_model": model_name,
                "question_text_clean": q_text,
                "nhan_chot_3_muc": row.get("nhan_chot_3_muc", ""),
                "nhan_chot_chi_tiet": row.get("nhan_chot_chi_tiet", ""),
                "reference_answer": row.get("reference_answer", ""),
                "retrieved_chunk_ids": [c.get("chunk_id") for c in top_chunks],
                "retrieved_contexts": [c.get("text", "") for c in top_chunks],
                "generated_answer": ans_text,
                "is_abstention": is_abstain,
                "n_words_answer": len(ans_text.split()),
                "usage": usage_meta,
            }
            results.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if idx % 30 == 0 or idx == len(ret_rows):
                print(f"[Gen {strategy}] {idx}/{len(ret_rows)} done", flush=True)

            if delay_sec > 0 and llm_call_fn is None and idx < len(ret_rows):
                time.sleep(delay_sec)

    return results


def cham_diem_judge_chien_luoc(
    strategy: str,
    rag_gen_path: Path,
    out_judge_path: Path,
    judge_model: str = "gemini-3.5-flash",
    batch_size: int = 4,
    delay_sec: float = 1.2,
    gemini_call_fn: Optional[Callable[[str, str], Union[Dict, List]]] = None,
) -> Dict:
    """Chấm điểm RAGAS bằng Giám khảo Gemini cho kết quả sinh của một chiến lược phân đoạn."""
    rag_rows, _ = doc_danh_sach_jsonl(rag_gen_path)
    out_judge_path.parent.mkdir(parents=True, exist_ok=True)
    call_fn = gemini_call_fn or (lambda prompt, m: goi_gemini_json(prompt, model_name=m))

    existing_rag: Dict[str, Dict] = {}
    if out_judge_path.exists():
        old_rag, _ = doc_danh_sach_jsonl(out_judge_path)
        for r in old_rag:
            if r.get("qid_moi") and "context_precision" in r:
                existing_rag[r["qid_moi"]] = r

    rag_to_judge = [r for r in rag_rows if r.get("qid_moi") not in existing_rag]
    mode_name = f"naive_rag_{strategy}"

    if rag_to_judge:
        with open(out_judge_path, "a", encoding="utf-8") as f_append:
            for start in range(0, len(rag_to_judge), batch_size):
                batch = rag_to_judge[start : start + batch_size]
                prompt = tao_prompt_cham_batch_rag(batch)
                res_list = call_fn(prompt, judge_model)
                batch_map: Dict[str, Dict] = {}
                if isinstance(res_list, dict):
                    res_list = res_list.get("items") or res_list.get("results") or [res_list]
                if isinstance(res_list, list):
                    for idx_b, obj in enumerate(res_list):
                        if isinstance(obj, dict):
                            q_key = str(
                                obj.get("qid_moi")
                                or (batch[idx_b]["qid_moi"] if idx_b < len(batch) else "")
                            )
                            if q_key:
                                batch_map[q_key] = obj

                for r in batch:
                    qid = r.get("qid_moi", "")
                    j = batch_map.get(qid, {})
                    rel_flags = j.get("chunk_relevance")
                    if not isinstance(rel_flags, list) or not rel_flags:
                        rel_flags = [0, 0, 0, 0, 0]
                    rel_flags = [1 if int(x) > 0 else 0 for x in rel_flags[:5]]
                    cp = tinh_context_precision_at_k(rel_flags)

                    is_abs = bool(r.get("is_abstention", False))
                    faith = 1.0 if is_abs else round(float(j.get("faithfulness", 1.0)), 4)
                    ans_rel = 0.0 if is_abs else round(float(j.get("answer_relevance", 0.8)), 4)
                    has_hallu = 0 if is_abs else int(j.get("has_hallucination", 0))
                    j_ans = int(j.get("judge_answerable", 1 if sum(rel_flags) > 0 else 0))

                    rec = {
                        "qid_moi": qid,
                        "asin": r.get("asin", ""),
                        "mode": mode_name,
                        "strategy": strategy,
                        "nhan_chot_3_muc": r.get("nhan_chot_3_muc", ""),
                        "nhan_chot_chi_tiet": r.get("nhan_chot_chi_tiet", ""),
                        "is_abstention": is_abs,
                        "generated_answer": r.get("generated_answer", ""),
                        "chunk_relevance": rel_flags,
                        "judge_answerable": j_ans,
                        "context_precision": cp,
                        "faithfulness": faith,
                        "answer_relevance": ans_rel,
                        "has_hallucination": has_hallu,
                    }
                    existing_rag[qid] = rec
                    f_append.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_append.flush()

                if delay_sec > 0 and gemini_call_fn is None:
                    time.sleep(delay_sec)

    judged_rows: List[Dict] = []
    with open(out_judge_path, "w", encoding="utf-8") as fout:
        for r in rag_rows:
            qid = r.get("qid_moi", "")
            rec = existing_rag[qid]
            judged_rows.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return tong_hop_ket_qua_judge_chien_luoc(strategy, judged_rows)


def tong_hop_ket_qua_judge_chien_luoc(strategy: str, judged_rows: Sequence[Dict]) -> Dict:
    """Tính toàn bộ chỉ số RAGAS, tỷ lệ trả lời theo nhóm và Cohen's kappa cho một chiến lược."""

    def _mean(vals: Sequence[float]) -> float:
        return round(float(statistics.mean(vals)), 4) if vals else 0.0

    direct = [r for r in judged_rows if not r["is_abstention"]]
    yes_grp = [r for r in judged_rows if r["nhan_chot_3_muc"] == "yes"]
    part_grp = [r for r in judged_rows if r["nhan_chot_3_muc"] == "partial"]
    ans_grp = [r for r in judged_rows if r["nhan_chot_3_muc"] in ("yes", "partial")]
    no_grp = [r for r in judged_rows if r["nhan_chot_3_muc"] == "no"]

    yes_direct = sum(1 for r in yes_grp if not r["is_abstention"])
    part_direct = sum(1 for r in part_grp if not r["is_abstention"])
    no_abstain = sum(1 for r in no_grp if r["is_abstention"])

    y_human_lenient = [1 if r["nhan_chot_3_muc"] in ("yes", "partial") else 0 for r in judged_rows]
    y_human_strict = [1 if r["nhan_chot_3_muc"] == "yes" else 0 for r in judged_rows]
    y_judge_ans = [int(r.get("judge_answerable", 0)) for r in judged_rows]
    y_rag_answered = [0 if r["is_abstention"] else 1 for r in judged_rows]

    by_qid: Dict[str, Dict] = {r["qid_moi"]: r for r in judged_rows}

    return {
        "strategy": strategy,
        "n_total": len(judged_rows),
        "n_direct": len(direct),
        "n_abstain": len(judged_rows) - len(direct),
        "yes_direct": yes_direct,
        "yes_total": len(yes_grp),
        "yes_answer_rate": round(yes_direct * 100.0 / max(1, len(yes_grp)), 2),
        "partial_direct": part_direct,
        "partial_total": len(part_grp),
        "partial_answer_rate": round(part_direct * 100.0 / max(1, len(part_grp)), 2),
        "no_abstain": no_abstain,
        "no_total": len(no_grp),
        "trap_rejection_rate": round(no_abstain * 100.0 / max(1, len(no_grp)), 2),
        "context_precision_yes": _mean([r["context_precision"] for r in yes_grp]),
        "context_precision_partial": _mean([r["context_precision"] for r in part_grp]),
        "context_precision_ans": _mean([r["context_precision"] for r in ans_grp]),
        "faithfulness_direct": _mean([r["faithfulness"] for r in direct]),
        "faithfulness_all": _mean([r["faithfulness"] for r in judged_rows]),
        "answer_relevance_direct": _mean([r["answer_relevance"] for r in direct]),
        "answer_relevance_yes": _mean([r["answer_relevance"] for r in yes_grp]),
        "answer_relevance_ans": _mean([r["answer_relevance"] for r in ans_grp]),
        "hallucination_count": sum(int(r["has_hallucination"]) for r in judged_rows),
        "hallucination_rate_direct": round(
            _mean([float(r["has_hallucination"]) for r in direct]) * 100.0, 2
        ),
        "kappa_judge_vs_strict": tinh_cohens_kappa(y_judge_ans, y_human_strict)["kappa"],
        "kappa_judge_vs_lenient": tinh_cohens_kappa(y_judge_ans, y_human_lenient)["kappa"],
        "kappa_judge_vs_rag": tinh_cohens_kappa(y_judge_ans, y_rag_answered)["kappa"],
        "by_qid": by_qid,
    }


def ghi_bao_cao_ragas_chien_luoc(
    summaries: Sequence[Dict],
    report_path: Path,
) -> None:
    """Xuất báo cáo Markdown đối sánh Generation & RAGAS giữa các chiến lược phân đoạn."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Đối sánh Sinh câu trả lời & RAGAS giữa các Chiến lược Phân đoạn (RQ1 & RQ2)",
        "",
        "## 1. Cấu hình hệ thống cố định",
        "- **Mô hình Embedding & Truy xuất:** `BAAI/bge-small-en-v1.5` + `ChromaDB` (Cosine, `ASIN-scoped Top-5`)",
        "- **Mô hình Sinh câu trả lời (Generator):** `qwen/qwen3.8-27b` (Groq API, `temperature = 0.0`, `max_tokens = 256`)",
        "- **Mô hình Giám khảo độc lập (Judge):** `gemini-3.5-flash` (Google Gemini API, `temperature = 0.0`)",
        "- **Tập đánh giá:** `eval_120_dev_labeled.jsonl` (**120 câu hỏi**: 27 `yes`, 64 `partial`, 29 `no`/bẫy)",
        "",
        "## 2. Hành vi Trả lời theo Nhóm Nhãn Chuẩn (`yes` / `partial` / `no`)",
        "",
        "| Chiến lược | Tổng trả lời trực tiếp (/120) | Tỷ lệ trả lời nhóm `yes` (/27) | Tỷ lệ trả lời nhóm `partial` (/64) | Tỷ lệ chặn câu bẫy `no` (/29) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| **`{s['strategy']}`** | {s['n_direct']}/120 ({s['n_direct']*100.0/max(1, s['n_total']):.2f}%) | "
            f"{s['yes_direct']}/{s['yes_total']} (**{s['yes_answer_rate']:.2f}%**) | "
            f"{s['partial_direct']}/{s['partial_total']} (**{s['partial_answer_rate']:.2f}%**) | "
            f"{s['no_abstain']}/{s['no_total']} (**{s['trap_rejection_rate']:.2f}%**) |"
        )

    lines.extend(
        [
            "",
            "## 3. Các Chỉ số RAGAS, Độ Trung thực, Ảo giác & Đồng thuận Giám khảo (Cohen's $\\kappa$)",
            "",
            "| Chiến lược | CP@5 (`yes`) | CP@5 (`yes+partial`) | Faithfulness (Direct) | Faithfulness (All) | Hallucination on Direct | Answer Rel (Direct) | Answer Rel (`yes+partial`) | $\\kappa$ (Judge vs. RAG) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in summaries:
        lines.append(
            f"| **`{s['strategy']}`** | {s['context_precision_yes']:.4f} | **{s['context_precision_ans']:.4f}** | "
            f"**{s['faithfulness_direct']:.4f}** | {s['faithfulness_all']:.4f} | "
            f"**{s['hallucination_rate_direct']:.2f}%** ({s['hallucination_count']}/{s['n_direct']}) | "
            f"{s['answer_relevance_direct']:.4f} | **{s['answer_relevance_ans']:.4f}** | "
            f"{s['kappa_judge_vs_rag']:.4f} |"
        )

    base = next((s for s in summaries if s["strategy"] == "fixed"), None)
    if base:
        lines.extend(
            [
                "",
                "## 4. Kiểm định McNemar so với Baseline `fixed` trên Nhóm `partial` (64 câu) & Hành vi Đúng Toàn tập (120 câu)",
                "",
                "| Chiến lược so với `fixed` | $\\Delta$ Câu trả lời `partial` | McNemar trên `partial` ($b / c$, $p$-value) | McNemar trên Hành vi đúng không ảo giác ($b / c$, $p$-value) |",
                "| :--- | ---: | :--- | :--- |",
            ]
        )
        qb = base["by_qid"]
        for s in summaries:
            if s["strategy"] == "fixed":
                continue
            qn = s["by_qid"]
            common = sorted(set(qb.keys()) & set(qn.keys()))
            part_q = [q for q in common if qb[q]["nhan_chot_3_muc"] == "partial"]
            pb = [0.0 if qb[q]["is_abstention"] else 1.0 for q in part_q]
            pn = [0.0 if qn[q]["is_abstention"] else 1.0 for q in part_q]
            mc_part = mcnemar_exact_pvalue(pb, pn)

            def _correct_behavior(rec: Dict) -> float:
                if rec["nhan_chot_3_muc"] == "no":
                    return 1.0 if rec["is_abstention"] else 0.0
                return 1.0 if (not rec["is_abstention"] and int(rec.get("has_hallucination", 0)) == 0) else 0.0

            cb = [_correct_behavior(qb[q]) for q in common]
            cn = [_correct_behavior(qn[q]) for q in common]
            mc_all = mcnemar_exact_pvalue(cb, cn)
            d_part = s["partial_direct"] - base["partial_direct"]
            lines.append(
                f"| `{s['strategy']}` vs. `fixed` | {d_part:+d} câu ({s['partial_answer_rate'] - base['partial_answer_rate']:+.2f}%) | "
                f"`b={mc_part['b_only_base']}, c={mc_part['c_only_new']}` ($p={mc_part['p_value']:.4f}$) | "
                f"`b={mc_all['b_only_base']}, c={mc_all['c_only_new']}` ($p={mc_all['p_value']:.4f}$) |"
            )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def chay_rag_va_judge_cac_chien_luoc(
    processed_dir: Path,
    report_path: Path,
    strategies: Sequence[str],
    generator_model: str = "qwen/qwen3.8-27b",
    judge_model: str = "gemini-3.5-flash",
    top_k: int = 5,
    batch_size: int = 5,
    gen_delay_sec: float = 0.4,
    judge_delay_sec: float = 0.8,
    skip_judge: bool = False,
    retrieval_prefix: str = "vector",
    run_tag: str = "",
    llm_call_fn: Optional[Callable[[Sequence[Dict[str, str]], str], Tuple[str, Dict]]] = None,
    gemini_call_fn: Optional[Callable[[str, str], Union[Dict, List]]] = None,
) -> Dict:
    """Chạy nối tiếp Sinh câu trả lời + Giám khảo RAGAS cho danh sách chiến lược phân đoạn."""
    summaries: List[Dict] = []
    ret_tag = f"_{retrieval_prefix}" if retrieval_prefix != "vector" else ""
    for st in strategies:
        ret_path = processed_dir / f"retrieval_{retrieval_prefix}_dev_{st}.jsonl"
        gen_path = processed_dir / f"generations_rag{ret_tag}_{st}_dev{run_tag}.jsonl"
        judge_path = processed_dir / f"eval_judge_rag{ret_tag}_{st}_dev{run_tag}.jsonl"

        sinh_cau_tra_loi_chien_luoc(
            strategy=st,
            retrieval_jsonl_path=ret_path,
            out_jsonl_path=gen_path,
            model_name=generator_model,
            top_k=top_k,
            delay_sec=gen_delay_sec,
            llm_call_fn=llm_call_fn,
        )
        if skip_judge:
            continue
        summary = cham_diem_judge_chien_luoc(
            strategy=st,
            rag_gen_path=gen_path,
            out_judge_path=judge_path,
            judge_model=judge_model,
            batch_size=batch_size,
            delay_sec=judge_delay_sec,
            gemini_call_fn=gemini_call_fn,
        )
        summaries.append(summary)
        print(
            f"[Judge {retrieval_prefix}/{st}{run_tag}] direct={summary['n_direct']}/120 "
            f"partial={summary['partial_direct']}/64 ({summary['partial_answer_rate']}%) "
            f"CP@5(ans)={summary['context_precision_ans']:.4f} "
            f"Faith(direct)={summary['faithfulness_direct']:.4f} "
            f"Hallu={summary['hallucination_rate_direct']}%",
            flush=True,
        )
        ghi_bao_cao_ragas_chien_luoc(summaries, report_path)

    if summaries:
        ghi_bao_cao_ragas_chien_luoc(summaries, report_path)
    return {
        "summaries": summaries,
        "report_path": str(report_path),
    }
