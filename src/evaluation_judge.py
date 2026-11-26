"""Module Giám khảo LLM (Gemini Flash) chấm điểm RAGAS (Faithfulness, Answer Relevance,
Context Precision) và kiểm định độ đồng thuận Cohen's kappa với nhãn người (RQ2 & RQ3).
"""

import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from src.label_verification import tinh_cohens_kappa
from src.retrieval_bm25 import doc_danh_sach_jsonl

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def normalize_judge_item(
    judge_obj: Optional[Dict],
    is_abstention: bool,
    item_id: str = "",
) -> Dict:
    """Chuẩn hóa điểm số giám khảo và áp dụng fallback đối xứng khi JSON lỗi hoặc thiếu khóa."""
    if is_abstention:
        return {
            "faithfulness": 1.0,
            "answer_relevance": 0.0,
            "has_hallucination": 0,
            "judge_parse_fallback": False,
        }

    if isinstance(judge_obj, dict) and judge_obj.get("faithfulness") is not None:
        return {
            "faithfulness": round(float(judge_obj["faithfulness"]), 4),
            "answer_relevance": round(float(judge_obj.get("answer_relevance", 0.0) or 0.0), 4),
            "has_hallucination": int(judge_obj.get("has_hallucination", 0) or 0),
            "judge_parse_fallback": False,
        }

    logger.warning(
        "Judge JSON rỗng hoặc thiếu khóa 'faithfulness' tại qid=%s; áp dụng fallback trung lập.",
        item_id or "unknown",
    )
    return {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "has_hallucination": 0,
        "judge_parse_fallback": True,
    }


def lay_gemini_api_key(dotenv_path: Optional[Path] = None) -> str:
    """Lấy GEMINI_API_KEY từ biến môi trường hoặc file .env."""
    env_path = dotenv_path or (ROOT / ".env")
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu biến môi trường GEMINI_API_KEY (hoặc chưa khai báo trong .env).")
    return api_key


def tinh_context_precision_at_k(relevance_flags: Sequence[int]) -> float:
    """Tính Context Precision@K theo công thức chuẩn của RAGAS.

    CP@K = (sum_{k=1..K} (Precision@k * r_k)) / (sum_{k=1..K} r_k)
    Trong đó r_k thuộc {0, 1} là nhãn liên quan của đoạn văn ở vị trí thứ k.
    """
    if not relevance_flags:
        return 0.0
    flags = [1 if int(x) > 0 else 0 for x in relevance_flags]
    total_rel = sum(flags)
    if total_rel == 0:
        return 0.0

    weighted_sum = 0.0
    rel_so_far = 0
    for idx, r_k in enumerate(flags, start=1):
        if r_k == 1:
            rel_so_far += 1
            weighted_sum += float(rel_so_far) / float(idx)

    return round(weighted_sum / float(total_rel), 4)


def _parse_json_an_toan(raw_text: str) -> Union[Dict, List]:
    cleaned = CODE_FENCE_RE.sub("", (raw_text or "").strip()).strip()
    return json.loads(cleaned)


def goi_gemini_json(
    prompt_text: str,
    model_name: str = "gemini-3.5-flash",
    api_key: Optional[str] = None,
    max_retries: int = 8,
) -> Union[Dict, List]:
    """Gọi Gemini API trả về cấu trúc JSON với cơ chế fallback model và retry tự động."""
    import httpx

    key = api_key or lay_gemini_api_key()
    fallback_models = [
        model_name,
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]
    candidate_models: List[str] = []
    for m in fallback_models:
        if m and m not in candidate_models:
            candidate_models.append(m)

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    last_err = ""
    for attempt in range(1, max_retries + 1):
        cur_model = candidate_models[(attempt - 1) % len(candidate_models)]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent?key={key}"
        try:
            resp = httpx.post(url, json=payload, timeout=75.0)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2.0 * attempt, 10.0))
            continue

        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates") or []
            if candidates:
                parts = (candidates[0].get("content") or {}).get("parts") or []
                for p in parts:
                    if "text" in p and p["text"].strip():
                        try:
                            return _parse_json_an_toan(p["text"])
                        except Exception as parse_exc:
                            last_err = f"JSONDecodeError: {parse_exc} | text={p['text'][:200]}"
            time.sleep(2.0)
            continue

        if resp.status_code == 429:
            time.sleep(min(6.0 * attempt, 35.0))
            last_err = f"HTTP 429 ({cur_model}): {resp.text[:200]}"
            continue

        if resp.status_code in (400, 404, 500, 502, 503, 504):
            time.sleep(min(2.5 * attempt, 15.0))
            last_err = f"HTTP {resp.status_code} ({cur_model}): {resp.text[:200]}"
            continue

        raise RuntimeError(f"Lỗi gọi Gemini API ({cur_model}, HTTP {resp.status_code}): {resp.text[:300]}")

    raise RuntimeError(f"Quá số lần thử lại Gemini API ({max_retries}): {last_err}")


def tao_prompt_cham_batch_rag(batch_items: Sequence[Dict]) -> str:
    """Tạo prompt chấm điểm RAGAS theo lô cho các câu trả lời Naive RAG."""
    items_payload = []
    for it in batch_items:
        chunks_full = list((it.get("retrieved_contexts") or [])[:5])
        items_payload.append(
            {
                "qid_moi": it.get("qid_moi", ""),
                "question": it.get("question_text_clean", ""),
                "reference_answer": it.get("reference_answer", ""),
                "retrieved_chunks": chunks_full,
                "generated_answer": it.get("generated_answer", ""),
                "is_abstention": bool(it.get("is_abstention", False)),
            }
        )

    return (
        "You are an expert scientific evaluator for an E-commerce RAG Question Answering system (RAGAS framework).\n"
        "Evaluate each item in the input JSON array and return a JSON array of objects with the exact same order and `qid_moi`.\n\n"
        "For each item, provide these exact fields:\n"
        "1. `qid_moi` (string): The question ID.\n"
        "2. `chunk_relevance` (array of 5 integers, each 0 or 1): For each of the 5 `retrieved_chunks`, output 1 if that chunk contains information helpful/relevant to answering `question`, else 0.\n"
        "3. `judge_answerable` (integer 0 or 1): Output 1 if the `retrieved_chunks` collectively contain partial or full information to answer `question`, or 0 if the chunks contain no relevant information to answer `question`.\n"
        "4. `faithfulness` (float between 0.0 and 1.0): Measures factual grounding in `retrieved_chunks`.\n"
        "   - If `is_abstention` is true (the model states the reviews do not contain enough information), output 1.0 because it did not fabricate any unsupported claims.\n"
        "   - If `is_abstention` is false, compute (number of claims in `generated_answer` supported by `retrieved_chunks`) / (total claims in `generated_answer`). Output 0.0 if claims are fabricated.\n"
        "5. `answer_relevance` (float between 0.0 and 1.0): Measures how well `generated_answer` addresses the user's `question`.\n"
        "   - If `is_abstention` is true, output 0.0 (since no answer was provided to the customer).\n"
        "   - If `is_abstention` is false, output a score from 0.0 to 1.0 reflecting how directly, clearly, and relevantly `generated_answer` answers `question`.\n"
        "6. `has_hallucination` (integer 0 or 1): Output 1 if `generated_answer` states specific facts not supported by `retrieved_chunks`, else 0.\n\n"
        f"Input items:\n{json.dumps(items_payload, ensure_ascii=False)}"
    )


def tao_prompt_cham_batch_no_rag_direct(batch_items: Sequence[Dict]) -> str:
    """Tạo prompt kiểm tra ảo giác cho các câu trả lời trực tiếp của nhánh No-RAG."""
    items_payload = []
    for it in batch_items:
        items_payload.append(
            {
                "qid_moi": it.get("qid_moi", ""),
                "asin": it.get("asin", ""),
                "question": it.get("question_text_clean", ""),
                "human_ground_truth_label": it.get("nhan_chot_3_muc", ""),
                "reference_answer": it.get("reference_answer", ""),
                "human_verification_note": it.get("ly_do_chot", "") or it.get("ghi_chu_kiem_tay_10_doan", ""),
                "product_review_snippets_sample": (it.get("review_snippets_clean") or [])[:5],
                "no_rag_generated_answer": it.get("generated_answer", ""),
            }
        )

    return (
        "You are an expert evaluator auditing an LLM answering Amazon Electronics questions WITHOUT RAG context (No-RAG baseline).\n"
        "For each item below, the LLM attempted to give a direct answer (`no_rag_generated_answer`) without seeing the product reviews.\n"
        "Compare `no_rag_generated_answer` against the actual product ground truth (`reference_answer`, `human_verification_note`, and `product_review_snippets_sample`).\n\n"
        "Return a JSON array of objects with these exact fields:\n"
        "1. `qid_moi` (string): The question ID.\n"
        "2. `faithfulness` (float 0.0 to 1.0): Fraction of claims in `no_rag_generated_answer` that are factually true and supported by the actual product evidence (`reference_answer` / reviews). If the LLM guessed wrong product specs, wrong product type, or unverified claims, score 0.0.\n"
        "3. `answer_relevance` (float 0.0 to 1.0): How relevantly and accurately the answer addresses the question (if the answer is a hallucination about the wrong product or wrong spec, score 0.0).\n"
        "4. `has_hallucination` (integer 0 or 1): Output 1 if `no_rag_generated_answer` contains fabricated, speculative, or false claims about the product, else 0.\n\n"
        f"Input items:\n{json.dumps(items_payload, ensure_ascii=False)}"
    )


def chay_danh_gia_judge_baseline(
    eval_labeled_path: Path,
    no_rag_gen_path: Path,
    rag_gen_path: Path,
    out_no_rag_judge_path: Path,
    out_rag_judge_path: Path,
    report_path: Path,
    judge_model: str = "gemini-3.5-flash",
    batch_size: int = 4,
    delay_sec: float = 1.5,
    gemini_call_fn: Optional[Callable[[str, str], Union[Dict, List]]] = None,
) -> Dict:
    """Thực thi Giám khảo Gemini chấm điểm Baseline 0 (No-RAG) và Baseline 1 (Naive RAG)."""
    eval_rows, _ = doc_danh_sach_jsonl(eval_labeled_path)
    no_rag_rows, _ = doc_danh_sach_jsonl(no_rag_gen_path)
    rag_rows, _ = doc_danh_sach_jsonl(rag_gen_path)

    eval_by_qid = {r["qid_moi"]: r for r in eval_rows if r.get("qid_moi")}
    call_fn = gemini_call_fn or (lambda prompt, m: goi_gemini_json(prompt, model_name=m))

    out_no_rag_judge_path.parent.mkdir(parents=True, exist_ok=True)
    existing_norag: Dict[str, Dict] = {}
    if out_no_rag_judge_path.exists():
        old_nr, _ = doc_danh_sach_jsonl(out_no_rag_judge_path)
        for r in old_nr:
            if r.get("qid_moi") and "faithfulness" in r:
                existing_norag[r["qid_moi"]] = r

    direct_to_judge: List[Dict] = []
    for r in no_rag_rows:
        qid = r.get("qid_moi", "")
        if qid in existing_norag:
            continue
        if not r.get("is_abstention", False):
            merged = dict(eval_by_qid.get(qid, {}))
            merged.update(r)
            direct_to_judge.append(merged)

    judged_direct_map: Dict[str, Dict] = {}
    for start in range(0, len(direct_to_judge), batch_size):
        batch = direct_to_judge[start : start + batch_size]
        prompt = tao_prompt_cham_batch_no_rag_direct(batch)
        res_list = call_fn(prompt, judge_model)
        if isinstance(res_list, dict):
            res_list = res_list.get("items") or res_list.get("results") or [res_list]
        if isinstance(res_list, list):
            for obj in res_list:
                if isinstance(obj, dict) and obj.get("qid_moi"):
                    judged_direct_map[str(obj["qid_moi"])] = obj
        if delay_sec > 0 and gemini_call_fn is None:
            time.sleep(delay_sec)

    no_rag_judged_rows: List[Dict] = []
    with open(out_no_rag_judge_path, "w", encoding="utf-8") as fout:
        for r in no_rag_rows:
            qid = r.get("qid_moi", "")
            if qid in existing_norag:
                rec = existing_norag[qid]
                no_rag_judged_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            if r.get("is_abstention", False):
                rec = {
                    "qid_moi": qid,
                    "asin": r.get("asin", ""),
                    "mode": "no_rag",
                    "nhan_chot_3_muc": r.get("nhan_chot_3_muc", ""),
                    "nhan_chot_chi_tiet": r.get("nhan_chot_chi_tiet", ""),
                    "is_abstention": True,
                    "generated_answer": r.get("generated_answer", ""),
                    "faithfulness": 1.0,
                    "answer_relevance": 0.0,
                    "context_precision": 0.0,
                    "has_hallucination": 0,
                }
            else:
                j = judged_direct_map.get(qid, {})
                norm = normalize_judge_item(j, is_abstention=False, item_id=qid)
                rec = {
                    "qid_moi": qid,
                    "asin": r.get("asin", ""),
                    "mode": "no_rag",
                    "nhan_chot_3_muc": r.get("nhan_chot_3_muc", ""),
                    "nhan_chot_chi_tiet": r.get("nhan_chot_chi_tiet", ""),
                    "is_abstention": False,
                    "generated_answer": r.get("generated_answer", ""),
                    "faithfulness": norm["faithfulness"],
                    "answer_relevance": norm["answer_relevance"],
                    "context_precision": 0.0,
                    "has_hallucination": norm["has_hallucination"],
                    "judge_parse_fallback": norm["judge_parse_fallback"],
                }
            no_rag_judged_rows.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    out_rag_judge_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rag: Dict[str, Dict] = {}
    if out_rag_judge_path.exists():
        old_rag, _ = doc_danh_sach_jsonl(out_rag_judge_path)
        for r in old_rag:
            if r.get("qid_moi") and "context_precision" in r:
                existing_rag[r["qid_moi"]] = r

    rag_to_judge = [r for r in rag_rows if r.get("qid_moi") not in existing_rag]

    if rag_to_judge:
        with open(out_rag_judge_path, "a", encoding="utf-8") as f_append:
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
                    norm = normalize_judge_item(j, is_abstention=is_abs, item_id=qid)
                    j_ans = int(j.get("judge_answerable", 1 if sum(rel_flags) > 0 else 0))

                    rec = {
                        "qid_moi": qid,
                        "asin": r.get("asin", ""),
                        "mode": "naive_rag_fixed",
                        "nhan_chot_3_muc": r.get("nhan_chot_3_muc", ""),
                        "nhan_chot_chi_tiet": r.get("nhan_chot_chi_tiet", ""),
                        "is_abstention": is_abs,
                        "generated_answer": r.get("generated_answer", ""),
                        "chunk_relevance": rel_flags,
                        "judge_answerable": j_ans,
                        "context_precision": cp,
                        "faithfulness": norm["faithfulness"],
                        "answer_relevance": norm["answer_relevance"],
                        "has_hallucination": norm["has_hallucination"],
                        "judge_parse_fallback": norm["judge_parse_fallback"],
                    }
                    existing_rag[qid] = rec
                    f_append.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_append.flush()

                if delay_sec > 0 and gemini_call_fn is None:
                    time.sleep(delay_sec)

    rag_judged_rows: List[Dict] = []
    with open(out_rag_judge_path, "w", encoding="utf-8") as fout:
        for r in rag_rows:
            qid = r.get("qid_moi", "")
            rec = existing_rag[qid]
            rag_judged_rows.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _mean(vals: Sequence[float]) -> float:
        return round(float(statistics.mean(vals)), 4) if vals else 0.0

    nr_direct = [r for r in no_rag_judged_rows if not r["is_abstention"]]
    nr_ans_group = [r for r in no_rag_judged_rows if r["nhan_chot_3_muc"] in ("yes", "partial")]
    nr_no_group = [r for r in no_rag_judged_rows if r["nhan_chot_3_muc"] == "no"]

    rag_direct = [r for r in rag_judged_rows if not r["is_abstention"]]
    rag_ans_group = [r for r in rag_judged_rows if r["nhan_chot_3_muc"] in ("yes", "partial")]
    rag_yes_group = [r for r in rag_judged_rows if r["nhan_chot_3_muc"] == "yes"]
    rag_no_group = [r for r in rag_judged_rows if r["nhan_chot_3_muc"] == "no"]

    summary_no_rag = {
        "n_total": len(no_rag_judged_rows),
        "n_direct": len(nr_direct),
        "n_abstain": len(no_rag_judged_rows) - len(nr_direct),
        "faithfulness_direct": _mean([r["faithfulness"] for r in nr_direct]),
        "faithfulness_all": _mean([r["faithfulness"] for r in no_rag_judged_rows]),
        "answer_relevance_direct": _mean([r["answer_relevance"] for r in nr_direct]),
        "answer_relevance_ans_group": _mean([r["answer_relevance"] for r in nr_ans_group]),
        "hallucination_rate_on_direct": _mean([float(r["has_hallucination"]) for r in nr_direct]),
        "hallucination_count_total": sum(int(r["has_hallucination"]) for r in no_rag_judged_rows),
        "trap_rejection_rate": _mean([1.0 if r["is_abstention"] else 0.0 for r in nr_no_group]),
    }

    summary_rag = {
        "n_total": len(rag_judged_rows),
        "n_direct": len(rag_direct),
        "n_abstain": len(rag_judged_rows) - len(rag_direct),
        "context_precision_all": _mean([r["context_precision"] for r in rag_judged_rows]),
        "context_precision_ans_group": _mean([r["context_precision"] for r in rag_ans_group]),
        "context_precision_yes_group": _mean([r["context_precision"] for r in rag_yes_group]),
        "faithfulness_direct": _mean([r["faithfulness"] for r in rag_direct]),
        "faithfulness_all": _mean([r["faithfulness"] for r in rag_judged_rows]),
        "answer_relevance_direct": _mean([r["answer_relevance"] for r in rag_direct]),
        "answer_relevance_ans_group": _mean([r["answer_relevance"] for r in rag_ans_group]),
        "answer_relevance_yes_group": _mean([r["answer_relevance"] for r in rag_yes_group]),
        "hallucination_rate_on_direct": _mean([float(r["has_hallucination"]) for r in rag_direct]),
        "hallucination_count_total": sum(int(r["has_hallucination"]) for r in rag_judged_rows),
        "trap_rejection_rate": _mean([1.0 if r["is_abstention"] else 0.0 for r in rag_no_group]),
    }

    y_human_lenient = [1 if r["nhan_chot_3_muc"] in ("yes", "partial") else 0 for r in rag_judged_rows]
    y_human_strict = [1 if r["nhan_chot_3_muc"] == "yes" else 0 for r in rag_judged_rows]
    y_judge_ans = [int(r.get("judge_answerable", 0)) for r in rag_judged_rows]
    y_rag_answered = [0 if r["is_abstention"] else 1 for r in rag_judged_rows]

    kappa_judge_vs_human_lenient = tinh_cohens_kappa(y_judge_ans, y_human_lenient)
    kappa_judge_vs_human_strict = tinh_cohens_kappa(y_judge_ans, y_human_strict)
    kappa_rag_vs_human_strict = tinh_cohens_kappa(y_rag_answered, y_human_strict)
    kappa_judge_vs_rag_behavior = tinh_cohens_kappa(y_judge_ans, y_rag_answered)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Đánh giá Baseline bằng Khung RAGAS & Giám khảo LLM (`Gemini Flash`)",
        "",
        "## 1. Cấu hình hệ thống Baseline & Khung đánh giá",
        "- **Mô hình Embedding:** `BAAI/bge-small-en-v1.5` (384 chiều, Cosine similarity trên `ChromaDB`)",
        "- **Chiến lược phân đoạn cơ sở (Baseline Chunking):** `Fixed-size` (200 từ/đoạn, overlap 40 từ -> 2.814 chunks trên tập Dev)",
        "- **Mô hình Sinh câu trả lời (Generator):** `qwen/qwen3.8-27b` (Groq API, `temperature = 0.0`)",
        f"- **Mô hình Giám khảo độc lập (LLM-as-a-Judge):** `{judge_model}` (Google Gemini API, `temperature = 0.0`)",
        f"- **Tập đánh giá đối chuẩn:** `eval_120_dev_labeled.jsonl` (**{len(rag_judged_rows)} câu hỏi** đã kiểm định tay 2 vòng: 27 câu `yes`, 64 câu `partial`, 29 câu `no`/bẫy)",
        "",
        "## 2. Kết quả Định lượng Đối sánh: Baseline 0 (`No-RAG`) vs. Baseline 1 (`Naive RAG — Fixed-size`)",
        "",
        "| Chỉ số đánh giá (RAGAS & Hallucination Metrics) | Baseline 0 (`No-RAG`) | Baseline 1 (`Naive RAG — Fixed-size`) | Mức cải thiện của RAG |",
        "| :--- | ---: | ---: | ---: |",
        f"| **1. Số câu trả lời trực tiếp (`Direct Answers` / 120 câu)** | {summary_no_rag['n_direct']} ({summary_no_rag['n_direct']*100.0/max(1, summary_no_rag['n_total']):.2f}%) | **{summary_rag['n_direct']} ({summary_rag['n_direct']*100.0/max(1, summary_rag['n_total']):.2f}%)** | **+{summary_rag['n_direct'] - summary_no_rag['n_direct']} câu** |",
        f"| **2. Faithfulness trung bình trên các câu trả lời trực tiếp** *(Độ trung thực)* | {summary_no_rag['faithfulness_direct']:.4f} | **{summary_rag['faithfulness_direct']:.4f}** | **{summary_rag['faithfulness_direct'] - summary_no_rag['faithfulness_direct']:+.4f}** |",
        f"| **3. Faithfulness toàn tập (120 câu, tính cả từ chối đúng)** | {summary_no_rag['faithfulness_all']:.4f} | **{summary_rag['faithfulness_all']:.4f}** | **{summary_rag['faithfulness_all'] - summary_no_rag['faithfulness_all']:+.4f}** |",
        f"| **4. Tỷ lệ Ảo giác khi đưa ra câu trả lời (`Hallucination Rate on Direct`)** | {summary_no_rag['hallucination_rate_on_direct']*100.0:.2f}% ({summary_no_rag['hallucination_count_total']}/{summary_no_rag['n_direct']} câu) | **{summary_rag['hallucination_rate_on_direct']*100.0:.2f}% ({summary_rag['hallucination_count_total']}/{summary_rag['n_direct']} câu)** | **{(summary_rag['hallucination_rate_on_direct'] - summary_no_rag['hallucination_rate_on_direct'])*100.0:+.2f}%** |",
        f"| **5. Answer Relevance trên các câu trả lời trực tiếp** | {summary_no_rag['answer_relevance_direct']:.4f} | **{summary_rag['answer_relevance_direct']:.4f}** | **{summary_rag['answer_relevance_direct'] - summary_no_rag['answer_relevance_direct']:+.4f}** |",
        f"| **6. Answer Relevance trên nhóm câu có đáp án đầy đủ (`yes`, 27 câu)** | 0.0000 | **{summary_rag['answer_relevance_yes_group']:.4f}** | **+{summary_rag['answer_relevance_yes_group']:.4f}** |",
        f"| **7. Answer Relevance trên toàn bộ 91 câu có thông tin (`yes` + `partial`)** | {summary_no_rag['answer_relevance_ans_group']:.4f} | **{summary_rag['answer_relevance_ans_group']:.4f}** | **{summary_rag['answer_relevance_ans_group'] - summary_no_rag['answer_relevance_ans_group']:+.4f}** |",
        f"| **8. Context Precision@5 trên nhóm `yes` (27 câu)** | 0.0000 | **{summary_rag['context_precision_yes_group']:.4f}** | **+{summary_rag['context_precision_yes_group']:.4f}** |",
        f"| **9. Context Precision@5 trên nhóm `yes` + `partial` (91 câu)** | 0.0000 | **{summary_rag['context_precision_ans_group']:.4f}** | **+{summary_rag['context_precision_ans_group']:.4f}** |",
        f"| **10. Tỷ lệ chặn đúng câu hỏi bẫy (`Trap Rejection Rate` trên 29 câu `no`)** | {summary_no_rag['trap_rejection_rate']*100.0:.2f}% | **{summary_rag['trap_rejection_rate']*100.0:.2f}%** | **+{(summary_rag['trap_rejection_rate'] - summary_no_rag['trap_rejection_rate'])*100.0:.2f}%** |",
        "",
        "## 3. Kiểm định Độ tin cậy của Giám khảo LLM so với Nhãn người (Cohen's $\\kappa$ — Trả lời RQ3)",
        "",
        "| Cặp đối chiếu (N = 120 câu hỏi) | Accuracy | Precision | Recall | F1-Score | $P_o$ | $P_e$ | **Cohen's $\\kappa$** |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **1. Giám khảo Gemini (`judge_answerable`) vs. Nhãn người Nghiêm ngặt (`yes`)** | {kappa_judge_vs_human_strict['accuracy']:.4f} | {kappa_judge_vs_human_strict['precision']:.4f} | {kappa_judge_vs_human_strict['recall']:.4f} | {kappa_judge_vs_human_strict['f1']:.4f} | {kappa_judge_vs_human_strict['po']:.4f} | {kappa_judge_vs_human_strict['pe']:.4f} | **{kappa_judge_vs_human_strict['kappa']:.4f}** |",
        f"| **2. Giám khảo Gemini (`judge_answerable`) vs. Nhãn người Mở rộng (`yes+partial`)** | {kappa_judge_vs_human_lenient['accuracy']:.4f} | {kappa_judge_vs_human_lenient['precision']:.4f} | {kappa_judge_vs_human_lenient['recall']:.4f} | {kappa_judge_vs_human_lenient['f1']:.4f} | {kappa_judge_vs_human_lenient['po']:.4f} | {kappa_judge_vs_human_lenient['pe']:.4f} | **{kappa_judge_vs_human_lenient['kappa']:.4f}** |",
        f"| **3. Hành vi trả lời của Naive RAG vs. Nhãn người Nghiêm ngặt (`yes`)** | {kappa_rag_vs_human_strict['accuracy']:.4f} | {kappa_rag_vs_human_strict['precision']:.4f} | {kappa_rag_vs_human_strict['recall']:.4f} | {kappa_rag_vs_human_strict['f1']:.4f} | {kappa_rag_vs_human_strict['po']:.4f} | {kappa_rag_vs_human_strict['pe']:.4f} | **{kappa_rag_vs_human_strict['kappa']:.4f}** |",
        f"| **4. Giám khảo Gemini (`judge_answerable`) vs. Hành vi trả lời của Naive RAG** | {kappa_judge_vs_rag_behavior['accuracy']:.4f} | {kappa_judge_vs_rag_behavior['precision']:.4f} | {kappa_judge_vs_rag_behavior['recall']:.4f} | {kappa_judge_vs_rag_behavior['f1']:.4f} | {kappa_judge_vs_rag_behavior['po']:.4f} | {kappa_judge_vs_rag_behavior['pe']:.4f} | **{kappa_judge_vs_rag_behavior['kappa']:.4f}** |",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "summary_no_rag": summary_no_rag,
        "summary_rag": summary_rag,
        "kappa_judge_vs_human_strict": kappa_judge_vs_human_strict,
        "kappa_judge_vs_human_lenient": kappa_judge_vs_human_lenient,
        "kappa_rag_vs_human_strict": kappa_rag_vs_human_strict,
        "kappa_judge_vs_rag_behavior": kappa_judge_vs_rag_behavior,
        "out_no_rag_judge_path": str(out_no_rag_judge_path),
        "out_rag_judge_path": str(out_rag_judge_path),
        "report_path": str(report_path),
    }
