"""Module sinh câu trả lời Baseline (No-RAG và Naive RAG) qua Groq API."""

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.retrieval_bm25 import doc_danh_sach_jsonl

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"(do not have enough information|don't have enough information|"
    r"do not contain enough information|does not contain enough information|"
    r"do not provide enough information|does not provide enough information|"
    r"do not have specific information|don't have specific information|"
    r"do not have access to|don't have access to|"
    r"cannot determine|can't determine|not enough context|not enough information|"
    r"insufficient information|unable to determine|cannot answer|can't answer|"
    r"not mentioned in the provided|not provided in the context|"
    r"no information provided|i don't know|i do not know)",
    re.IGNORECASE,
)

SYSTEM_PROMPT_NO_RAG = (
    "You are an e-commerce product question answering assistant for Amazon Electronics. "
    "Answer the customer's question about the specified product (ASIN) concisely in 1 to 3 sentences "
    "based on your knowledge. If you do not know the answer or lack specific product details, "
    "state clearly that you do not have enough information."
)

SYSTEM_PROMPT_RAG = (
    "You are an e-commerce product question answering assistant for Amazon Electronics. "
    "Answer the customer's question concisely in 1 to 3 sentences using ONLY the provided product review snippets. "
    "Do not invent facts or extrapolate beyond the context. "
    "If the provided snippets do not contain enough information to answer the question, "
    "state clearly that the provided product reviews do not contain enough information."
)


def lay_llm_config(
    dotenv_path: Optional[Path] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """Lấy (base_url, api_key) từ tham số, biến môi trường hoặc file .env.

    Mặc định dùng Groq API; nếu đặt LLM_BASE_URL (ví dụ Ollama/vLLM trên Colab)
    thì dùng endpoint OpenAI-compatible đó và không bắt buộc GROQ_API_KEY.
    """
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

    resolved_url = (base_url or os.environ.get("LLM_BASE_URL", "")).strip()
    if not resolved_url:
        resolved_url = "https://api.groq.com/openai/v1"
    resolved_url = resolved_url.rstrip("/")

    is_groq = "api.groq.com" in resolved_url
    resolved_key = (
        api_key
        or os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("GROQ_API_KEY", "").strip()
    )
    if not resolved_key:
        if is_groq:
            raise RuntimeError("Thiếu biến môi trường GROQ_API_KEY (hoặc chưa khai báo trong .env).")
        resolved_key = "local"

    return resolved_url, resolved_key


def lay_groq_api_key(dotenv_path: Optional[Path] = None) -> str:
    """Lấy GROQ_API_KEY từ biến môi trường hoặc file .env."""
    _, key = lay_llm_config(dotenv_path=dotenv_path, base_url="https://api.groq.com/openai/v1")
    return key


def lam_sach_dau_ra_llm(raw_text: str) -> str:
    """Loại bỏ thẻ <think>...</think> (nếu có) và chuẩn hóa khoảng trắng đầu/cuối."""
    if not isinstance(raw_text, str):
        return ""
    cleaned = THINK_TAG_RE.sub("", raw_text)
    return cleaned.strip()


def la_cau_tu_choi_tra_loi(answer_text: str) -> bool:
    """Kiểm tra xem câu trả lời của LLM có phải là lời từ chối do thiếu thông tin hay không."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        return True
    return bool(ABSTAIN_RE.search(answer_text))


def tao_prompt_no_rag(asin: str, category: str, question_text: str) -> List[Dict[str, str]]:
    """Tạo danh sách messages cho kịch bản Baseline 0 (No-RAG)."""
    user_msg = (
        f"Product ASIN: {asin} (Category: {category or 'Electronics'})\n"
        f"Customer Question: {question_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
        {"role": "user", "content": user_msg},
    ]


def tao_prompt_rag(
    asin: str,
    category: str,
    question_text: str,
    retrieved_chunks: Sequence[Dict],
) -> List[Dict[str, str]]:
    """Tạo danh sách messages cho kịch bản Baseline 1 (Naive RAG với Top-k chunks)."""
    ctx_blocks: List[str] = []
    for idx, ch in enumerate(retrieved_chunks, start=1):
        ctext = (ch.get("text") or "").strip()
        cid = ch.get("chunk_id") or f"chunk_{idx}"
        ctx_blocks.append(f"[Snippet {idx} | {cid}]\n{ctext}")

    context_str = "\n\n".join(ctx_blocks) if ctx_blocks else "(No context available)"
    user_msg = (
        f"Product ASIN: {asin} (Category: {category or 'Electronics'})\n\n"
        f"Retrieved Product Review Snippets:\n{context_str}\n\n"
        f"Customer Question: {question_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": user_msg},
    ]


def goi_groq_chat_completion(
    messages: Sequence[Dict[str, str]],
    model_name: str = "qwen/qwen3.8-27b",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 12,
    base_url: Optional[str] = None,
    timeout_sec: float = 120.0,
) -> Tuple[str, Dict]:
    """Gọi OpenAI-compatible Chat Completions API (Groq hoặc Colab/Ollama/vLLM)."""
    import httpx

    resolved_base, key = lay_llm_config(base_url=base_url, api_key=api_key)
    url = f"{resolved_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "api.groq.com" not in resolved_base:
        payload["reasoning_effort"] = "none"

    last_err = ""
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout_sec)
        except Exception as exc:
            last_err = str(exc)
            time.sleep(min(2.0 * attempt, 10.0))
            continue

        latency = round(time.time() - t0, 3)
        if resp.status_code == 200:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = lam_sach_dau_ra_llm(msg.get("content") or "")
            usage = data.get("usage") or {}
            rem_tokens_str = resp.headers.get("x-ratelimit-remaining-tokens")
            if rem_tokens_str and rem_tokens_str.isdigit():
                if int(rem_tokens_str) < 1800:
                    time.sleep(6.5)
            meta = {
                "latency_sec": latency,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
            return content, meta

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            wait_s = (
                float(retry_after) + 1.5
                if (retry_after and retry_after.replace(".", "").isdigit())
                else min(8.0 * attempt, 35.0)
            )
            time.sleep(min(wait_s, 45.0))
            last_err = f"HTTP 429: {resp.text[:200]}"
            continue

        if resp.status_code >= 500:
            time.sleep(2.5 * attempt)
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            continue

        raise RuntimeError(f"Lỗi gọi Groq API (HTTP {resp.status_code}): {resp.text[:300]}")

    raise RuntimeError(f"Quá số lần thử lại Groq API ({max_retries}): {last_err}")


def chay_generation_no_rag_baseline(
    eval_questions_path: Path,
    out_jsonl_path: Path,
    report_path: Path,
    model_name: str = "qwen/qwen3.8-27b",
    delay_sec: float = 0.8,
    llm_call_fn: Optional[Callable[[Sequence[Dict[str, str]], str], Tuple[str, Dict]]] = None,
) -> Dict:
    """Thực thi Baseline 0 (No-RAG) cho tập 120 câu hỏi đánh giá, hỗ trợ resume."""
    questions, n_corrupt_q = doc_danh_sach_jsonl(eval_questions_path)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    existing_by_qid: Dict[str, Dict] = {}
    if out_jsonl_path.exists():
        old_rows, _ = doc_danh_sach_jsonl(out_jsonl_path)
        for r in old_rows:
            qid = r.get("qid_moi")
            if qid and r.get("generated_answer"):
                existing_by_qid[qid] = r

    call_fn = llm_call_fn or (lambda msgs, m: goi_groq_chat_completion(msgs, model_name=m))

    results: List[Dict] = []
    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for idx, q in enumerate(questions, start=1):
            qid = q.get("qid_moi", "")
            if qid in existing_by_qid:
                rec = dict(existing_by_qid[qid])
                rec["is_abstention"] = la_cau_tu_choi_tra_loi(rec.get("generated_answer", ""))
                results.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            asin = q.get("asin", "")
            doc_id = q.get("doc_id", "")
            q_text = q.get("question_text_clean") or q.get("question_text") or ""
            msgs = tao_prompt_no_rag(asin=asin, category="Electronics", question_text=q_text)

            ans_text, usage_meta = call_fn(msgs, model_name)
            is_abstain = la_cau_tu_choi_tra_loi(ans_text)

            rec = {
                "qid_moi": qid,
                "asin": asin,
                "doc_id": doc_id,
                "mode": "no_rag",
                "generator_model": model_name,
                "question_text_clean": q_text,
                "question_type": q.get("question_type", ""),
                "nhan_chot_3_muc": q.get("nhan_chot_3_muc", ""),
                "nhan_chot_chi_tiet": q.get("nhan_chot_chi_tiet", ""),
                "reference_answer": q.get("reference_answer", ""),
                "generated_answer": ans_text,
                "is_abstention": is_abstain,
                "n_words_answer": len(ans_text.split()),
                "usage": usage_meta,
            }
            results.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if delay_sec > 0 and llm_call_fn is None and idx < len(questions):
                time.sleep(delay_sec)

    n_total = len(results)
    n_abstain = sum(1 for r in results if r.get("is_abstention"))
    n_direct = n_total - n_abstain

    by_label_3: Dict[str, Dict[str, int]] = {}
    for r in results:
        lbl = r.get("nhan_chot_3_muc", "unknown")
        grp = by_label_3.setdefault(lbl, {"total": 0, "abstain": 0, "direct": 0})
        grp["total"] += 1
        if r.get("is_abstention"):
            grp["abstain"] += 1
        else:
            grp["direct"] += 1

    latencies = [float(r.get("usage", {}).get("latency_sec", 0.0)) for r in results if r.get("usage")]
    total_tokens = sum(int(r.get("usage", {}).get("total_tokens", 0)) for r in results)
    median_words = round(float(statistics.median([r["n_words_answer"] for r in results])), 1) if results else 0.0
    mean_latency = round(float(statistics.mean(latencies)), 3) if latencies else 0.0

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Sinh câu trả lời Đối chứng Không Ngữ cảnh (Baseline 0: No-RAG)",
        "",
        "## 1. Cấu hình thực nghiệm",
        "- Chế độ: **`no_rag` (LLM-only, không cung cấp đoạn đánh giá sản phẩm)**",
        f"- Mô hình Generator: **`{model_name}`** (Groq API, `temperature = 0.0`, `max_tokens = 256`)",
        f"- Tập câu hỏi đánh giá: `{eval_questions_path.name}` (**{n_total} câu hỏi**)",
        f"- Thời gian phản hồi trung bình: **{mean_latency} giây / câu**",
        f"- Tổng số token tiêu thụ: **{total_tokens} tokens** (Độ dài câu trả lời trung vị: **{median_words} từ**)",
        "",
        "## 2. Thống kê hành vi trả lời của LLM khi Không có RAG (Phục vụ RQ2)",
        "",
        "| Nhãn thực tế trong tài liệu (`nhan_chot_3_muc`) | Tổng số câu | LLM tự trả lời trực tiếp (`Direct Answer`) | LLM từ chối trả lời (`Abstention`) | Tỷ lệ tự trả lời không cần ngữ cảnh |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for lbl in ("yes", "partial", "no"):
        if lbl in by_label_3:
            g = by_label_3[lbl]
            rate = (g["direct"] * 100.0 / g["total"]) if g["total"] else 0.0
            lines.append(
                f"| **`{lbl}`** | {g['total']} | {g['direct']} | {g['abstain']} | {rate:.2f}% |"
            )
    rate_all = (n_direct * 100.0 / n_total) if n_total else 0.0
    lines.extend(
        [
            f"| **Tổng cộng toàn tập** | **{n_total}** | **{n_direct}** | **{n_abstain}** | **{rate_all:.2f}%** |",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_questions": n_total,
        "n_corrupt_questions": n_corrupt_q,
        "n_direct_answer": n_direct,
        "n_abstention": n_abstain,
        "by_label_3": by_label_3,
        "mean_latency_sec": mean_latency,
        "total_tokens": total_tokens,
        "out_jsonl_path": str(out_jsonl_path),
        "report_path": str(report_path),
    }


def chay_generation_rag_baseline(
    retrieval_jsonl_path: Path,
    out_jsonl_path: Path,
    report_path: Path,
    no_rag_jsonl_path: Optional[Path] = None,
    model_name: str = "qwen/qwen3.8-27b",
    top_k: int = 5,
    delay_sec: float = 1.2,
    llm_call_fn: Optional[Callable[[Sequence[Dict[str, str]], str], Tuple[str, Dict]]] = None,
) -> Dict:
    """Thực thi Baseline 1 (Naive RAG — Fixed-size + BGE-small-en-v1.5) cho 120 câu hỏi đánh giá."""
    ret_rows, n_corrupt = doc_danh_sach_jsonl(retrieval_jsonl_path)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    existing_by_qid: Dict[str, Dict] = {}
    if out_jsonl_path.exists():
        old_rows, _ = doc_danh_sach_jsonl(out_jsonl_path)
        for r in old_rows:
            qid = r.get("qid_moi")
            if qid and r.get("generated_answer"):
                existing_by_qid[qid] = r

    call_fn = llm_call_fn or (lambda msgs, m: goi_groq_chat_completion(msgs, model_name=m))

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
                "mode": "naive_rag_fixed",
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

            if delay_sec > 0 and llm_call_fn is None and idx < len(ret_rows):
                time.sleep(delay_sec)

    n_total = len(results)
    n_abstain = sum(1 for r in results if r.get("is_abstention"))
    n_direct = n_total - n_abstain

    by_label_3: Dict[str, Dict[str, int]] = {}
    for r in results:
        lbl = r.get("nhan_chot_3_muc", "unknown")
        grp = by_label_3.setdefault(lbl, {"total": 0, "abstain": 0, "direct": 0})
        grp["total"] += 1
        if r.get("is_abstention"):
            grp["abstain"] += 1
        else:
            grp["direct"] += 1

    latencies = [float(r.get("usage", {}).get("latency_sec", 0.0)) for r in results if r.get("usage")]
    total_tokens = sum(int(r.get("usage", {}).get("total_tokens", 0)) for r in results)
    median_words = round(float(statistics.median([r["n_words_answer"] for r in results])), 1) if results else 0.0
    mean_latency = round(float(statistics.mean(latencies)), 3) if latencies else 0.0

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Sinh câu trả lời Cơ sở Có Ngữ cảnh (Baseline 1: Naive RAG — Fixed-size)",
        "",
        "## 1. Cấu hình thực nghiệm",
        "- Chế độ: **`naive_rag_fixed` (Fixed-size 200 từ + `BAAI/bge-small-en-v1.5` + `ChromaDB` Top-5)**",
        f"- Mô hình Generator: **`{model_name}`** (Groq API, `temperature = 0.0`, `max_tokens = 256`)",
        f"- Tập dữ liệu truy xuất đầu vào: `{retrieval_jsonl_path.name}` (**{n_total} câu hỏi**)",
        f"- Thời gian phản hồi trung bình: **{mean_latency} giây / câu**",
        f"- Tổng số token tiêu thụ: **{total_tokens} tokens** (Độ dài câu trả lời trung vị: **{median_words} từ**)",
        "",
        "## 2. Thống kê hành vi trả lời của LLM khi Có RAG (Baseline 1)",
        "",
        "| Nhãn thực tế trong tài liệu (`nhan_chot_3_muc`) | Tổng số câu | LLM trả lời trực tiếp (`Direct Answer`) | LLM từ chối trả lời (`Abstention`) | Tỷ lệ cung cấp câu trả lời (`Answer Rate`) | Tỷ lệ từ chối đúng câu bẫy (`Rejection Rate`) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lbl in ("yes", "partial", "no"):
        if lbl in by_label_3:
            g = by_label_3[lbl]
            ans_rate = (g["direct"] * 100.0 / g["total"]) if g["total"] else 0.0
            rej_rate = (g["abstain"] * 100.0 / g["total"]) if g["total"] else 0.0
            lines.append(
                f"| **`{lbl}`** | {g['total']} | {g['direct']} | {g['abstain']} | {ans_rate:.2f}% | {rej_rate:.2f}% |"
            )
    rate_all = (n_direct * 100.0 / n_total) if n_total else 0.0
    rej_all = (n_abstain * 100.0 / n_total) if n_total else 0.0
    lines.extend(
        [
            f"| **Tổng cộng toàn tập** | **{n_total}** | **{n_direct}** | **{n_abstain}** | **{rate_all:.2f}%** | **{rej_all:.2f}%** |",
            "",
        ]
    )

    if no_rag_jsonl_path is not None and no_rag_jsonl_path.exists():
        norag_rows, _ = doc_danh_sach_jsonl(no_rag_jsonl_path)
        nr_by_lbl: Dict[str, Dict[str, int]] = {}
        for r in norag_rows:
            lbl = r.get("nhan_chot_3_muc", "unknown")
            grp = nr_by_lbl.setdefault(lbl, {"total": 0, "abstain": 0, "direct": 0})
            grp["total"] += 1
            if r.get("is_abstention"):
                grp["abstain"] += 1
            else:
                grp["direct"] += 1
        lines.extend(
            [
                "## 3. Đối sánh trực tiếp: Baseline 0 (No-RAG) vs. Baseline 1 (Naive RAG — Fixed-size)",
                "",
                "| Nhóm câu hỏi | Chỉ số | Baseline 0 (`No-RAG`) | Baseline 1 (`Naive RAG`) | Mức thay đổi |",
                "| :--- | :--- | ---: | ---: | ---: |",
            ]
        )
        for lbl, desc in (
            ("yes", "Câu có đáp án đầy đủ (`yes`, 27 câu)"),
            ("partial", "Câu có thông tin một phần (`partial`, 64 câu)"),
            ("no", "Câu không có thông tin / Bẫy (`no`, 29 câu)"),
        ):
            if lbl in nr_by_lbl and lbl in by_label_3:
                g0 = nr_by_lbl[lbl]
                g1 = by_label_3[lbl]
                r0 = (g0["direct"] * 100.0 / g0["total"]) if g0["total"] else 0.0
                r1 = (g1["direct"] * 100.0 / g1["total"]) if g1["total"] else 0.0
                diff = r1 - r0
                sign = "+" if diff >= 0 else ""
                lines.append(
                    f"| **{desc}** | Tỷ lệ trả lời trực tiếp | {g0['direct']}/{g0['total']} ({r0:.2f}%) | **{g1['direct']}/{g1['total']} ({r1:.2f}%)** | **{sign}{diff:.2f}%** |"
                )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_questions": n_total,
        "n_corrupt": n_corrupt,
        "n_direct_answer": n_direct,
        "n_abstention": n_abstain,
        "by_label_3": by_label_3,
        "mean_latency_sec": mean_latency,
        "total_tokens": total_tokens,
        "out_jsonl_path": str(out_jsonl_path),
        "report_path": str(report_path),
    }
