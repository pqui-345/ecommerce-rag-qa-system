"""CLI demo kịch bản trình bày trực tiếp hệ thống RAG cho thương mại điện tử."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.clean import clean_text
from src.generation_baseline import (
    goi_groq_chat_completion,
    la_cau_tu_choi_tra_loi,
    lay_llm_config,
    tao_prompt_no_rag,
    tao_prompt_rag,
)
from src.retrieval_bm25 import BM25TinhDiem, bo_stopword, tokenize
from src.specialized_eval import SYSTEM_PROMPT_POLICY

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Đọc tệp JSONL theo luồng, bỏ qua dòng trống hoặc dòng JSON hỏng."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                logger.warning("Bỏ qua dòng JSON hỏng tại %s:%d (%s)", path.name, line_no, exc)
    return records


def load_jsonl_index(path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
    """Đọc tệp JSONL theo luồng và lập từ điển tra cứu theo khóa."""
    index: Dict[str, Dict[str, Any]] = {}
    for record in load_jsonl_records(path):
        key = record.get(key_field)
        if key is not None:
            index[str(key)] = record
    return index


def load_shopify_passages(path: Path) -> List[Dict[str, Any]]:
    """Trích danh sách đoạn văn phẳng từ kho chính sách Shopify Legal."""
    passages: List[Dict[str, Any]] = []
    for doc in load_jsonl_records(path):
        doc_id = doc.get("doc_id", "")
        title = doc.get("title", "")
        for p in doc.get("passages", []):
            if isinstance(p, dict):
                passages.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "passage_id": p.get("passage_id", ""),
                        "text": p.get("text", ""),
                    }
                )
    return passages


def search_bm25_chunks(
    chunks: Sequence[Dict[str, Any]],
    query: str,
    top_k: int = 3,
    asin: Optional[str] = None,
    text_field: str = "text",
) -> Tuple[List[Tuple[Dict[str, Any], float]], float]:
    """Truy xuất Top-K đoạn văn bằng Okapi BM25 và trả về độ trễ thực thi (ms)."""
    candidates = [c for c in chunks if asin is None or c.get("asin") == asin]
    if not candidates or top_k <= 0:
        return [], 0.0

    docs_tokens = [bo_stopword(tokenize(str(c.get(text_field, "")))) for c in candidates]
    query_tokens = bo_stopword(tokenize(query))

    start = time.perf_counter()
    bm25 = BM25TinhDiem(docs_tokens)
    scored = [(chunk, bm25.diem(query_tokens, idx)) for idx, chunk in enumerate(candidates)]
    scored.sort(key=lambda item: item[1], reverse=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return scored[:top_k], elapsed_ms


def summarize_test_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Tổng hợp các chỉ số đánh giá từ tệp kết quả tập test."""
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "direct": 0,
            "faithfulness_direct": 0.0,
            "cp5_answerable": 0.0,
            "answer_relevance_all": 0.0,
            "hallucinations": 0,
            "hallucination_rate_pct": 0.0,
        }

    direct_rows = [r for r in records if not r.get("is_abstention", False)]
    answerable_rows = [r for r in records if r.get("judge_answerable") == 1]
    hallucinations = sum(int(r.get("has_hallucination", 0)) for r in direct_rows)
    n_direct = len(direct_rows)

    faith_direct = (
        sum(float(r.get("faithfulness", 0.0)) for r in direct_rows) / n_direct
        if n_direct > 0
        else 0.0
    )
    cp5 = (
        sum(float(r.get("context_precision", 0.0)) for r in answerable_rows) / len(answerable_rows)
        if answerable_rows
        else 0.0
    )
    ans_rel = sum(float(r.get("answer_relevance", 0.0)) for r in records) / total
    hallu_rate = (hallucinations / n_direct * 100.0) if n_direct > 0 else 0.0

    return {
        "total": total,
        "direct": n_direct,
        "faithfulness_direct": faith_direct,
        "cp5_answerable": cp5,
        "answer_relevance_all": ans_rel,
        "hallucinations": hallucinations,
        "hallucination_rate_pct": hallu_rate,
    }


def _shorten(text: str, limit: int = 180) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def build_live_messages(
    query: str,
    source: str,
    hits: Sequence[Tuple[Dict[str, Any], float]],
    asin: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Tạo cặp prompt No-RAG và RAG (hoặc Policy RAG) cho chế độ hỏi đáp trực tiếp."""
    contexts = [chunk for chunk, _ in hits]
    if source == "shopify":
        no_rag_msgs = [
            {
                "role": "system",
                "content": (
                    "You are an e-commerce policy QA assistant for Shopify Legal Policies. "
                    "Answer concisely in 1 to 3 sentences based on your knowledge. "
                    "If you are not sure, state clearly that you do not have enough information."
                ),
            },
            {"role": "user", "content": f"Policy Question: {query}"},
        ]
        blocks = [
            f"[Passage {i} | {ch.get('passage_id') or ch.get('chunk_id') or f'p{i}'}]\n{(ch.get('text') or '').strip()}"
            for i, ch in enumerate(contexts, start=1)
        ]
        ctx_str = "\n\n".join(blocks) if blocks else "(No policy passages retrieved)"
        rag_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_POLICY},
            {"role": "user", "content": f"Retrieved Policy Passages:\n{ctx_str}\n\nQuestion: {query}"},
        ]
        return no_rag_msgs, rag_msgs

    target_asin = asin or (contexts[0].get("asin") if contexts else "UNKNOWN")
    no_rag_msgs = tao_prompt_no_rag(str(target_asin), "Electronics", query)
    rag_msgs = tao_prompt_rag(str(target_asin), "Electronics", query, contexts)
    return no_rag_msgs, rag_msgs


def _load_dotenv_into_env(dotenv_path: Optional[Path] = None) -> None:
    env_path = dotenv_path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def generate_live_answer(
    messages: Sequence[Dict[str, str]],
    model_name: str = "qwen/qwen3.8-27b",
    timeout_sec: float = 20.0,
) -> Tuple[Optional[str], str]:
    """Gọi LLM trực tiếp (ưu tiên Groq/OpenAI-compatible, dự phòng Gemini API)."""
    _load_dotenv_into_env()
    try:
        lay_llm_config()
        text, meta = goi_groq_chat_completion(
            messages,
            model_name=model_name,
            max_retries=1,
            timeout_sec=timeout_sec,
        )
        return text, f"Groq ({model_name}, {meta.get('latency_sec', 0):.2f}s)"
    except Exception as groq_exc:
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not gemini_key:
            return None, f"Offline/Khong co API key ({groq_exc})"
        try:
            import httpx

            sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
            user_parts = [m["content"] for m in messages if m.get("role") != "system"]
            payload: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
            }
            if sys_parts:
                payload["systemInstruction"] = {"parts": [{"text": "\n".join(sys_parts)}]}
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={gemini_key}"
            )
            t0 = time.perf_counter()
            resp = httpx.post(url, json=payload, timeout=timeout_sec)
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                data = resp.json()
                cands = data.get("candidates") or []
                parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
                ans = "".join(p.get("text", "") for p in parts).strip()
                if ans:
                    return ans, f"Gemini API fallback ({elapsed:.2f}s)"
            return None, f"API fallback HTTP {resp.status_code}"
        except Exception as gem_exc:
            return None, f"Offline/Loi mang ({gem_exc})"


def run_live_qa(
    query: str,
    source: str = "amazon",
    asin: Optional[str] = None,
    top_k: int = 5,
    model_name: str = "qwen/qwen3.8-27b",
    generator_fn: Callable[[Sequence[Dict[str, str]], str], Tuple[Optional[str], str]] = generate_live_answer,
) -> None:
    """Chạy truy xuất BM25 trên CPU kết hợp sinh câu trả lời trực tiếp No-RAG vs RAG."""
    _print_header("HOI DAP TRUC TIEP (LIVE RAG DEMO)")
    if source == "shopify":
        corpus_path = ROOT / "data" / "processed" / "policy_docs_shopify.jsonl"
        records = load_shopify_passages(corpus_path)
        if not records:
            print(f"- Khong tim thay {corpus_path.relative_to(ROOT)}.")
            return
        hits, ms = search_bm25_chunks(records, query, top_k=top_k)
        print(f"- Nguon: Shopify Legal (527 passages) | Truy xuat BM25: {ms:.2f} ms")
    else:
        corpus_path = ROOT / "data" / "processed" / "chunks_dev_recursive.jsonl"
        records = load_jsonl_records(corpus_path)
        if not records:
            print(f"- Khong tim thay {corpus_path.relative_to(ROOT)}.")
            return
        hits, ms = search_bm25_chunks(records, query, top_k=top_k, asin=asin)
        scope = f"ASIN={asin}" if asin else "Toan kho Global"
        print(f"- Nguon: Amazon Electronics ({scope}) | Truy xuat BM25: {ms:.2f} ms")

    for rank, (chunk, score) in enumerate(hits, start=1):
        cid = chunk.get("passage_id") or chunk.get("chunk_id")
        print(f"  #{rank} [{cid}] score={score:.3f} | {_shorten(chunk.get('text', ''), 145)}")

    no_rag_msgs, rag_msgs = build_live_messages(query, source, hits, asin=asin)
    ans_no_rag, info_0 = generator_fn(no_rag_msgs, model_name)
    ans_rag, info_1 = generator_fn(rag_msgs, model_name)

    if ans_no_rag is None and ans_rag is None:
        print(f"\n[Che do du phong Offline] Khong goi duoc LLM ({info_1}).")
        print("-> Da hien thi Top-K bang chung truy xuat o tren; chay '--step 3' de xem ca mau da cham.")
        return

    print(f"\n[Ket qua sinh truc tiep]")
    print(f"  No-RAG ({info_0}) [Tu choi={la_cau_tu_choi_tra_loi(ans_no_rag or '')}]:")
    print(f"    {ans_no_rag or '(Khong co phan hoi)'}")
    print(f"  RAG ({info_1}) [Tu choi={la_cau_tu_choi_tra_loi(ans_rag or '')}]:")
    print(f"    {ans_rag or '(Khong co phan hoi)'}")


def _print_header(title: str) -> None:
    print(f"\n[{title}]")


def run_step_1_clean() -> None:
    """Bước 1: Minh họa chuẩn hóa lũy đẳng clean_text bảo toàn ký hiệu kỹ thuật."""
    _print_header("BUOC 1: TIEN XU LY LUY DANG (clean_text)")
    raw_sample = (
        "Works with 15&#34; laptop, &amp;&lt;10 hours runtime<br>"
        "Size: 5*3 inches\u200b\n\n\nSupports >90% chargers."
    )
    once = clean_text(raw_sample)
    twice = clean_text(once)
    print(f"- Dau vao tho : {raw_sample!r}")
    print(f"- Sau clean x1: {once!r}")
    print(f"- Luy dang x2 : {once == twice} | Bao toan '5*3' & '<10 hours': {'5*3' in once and '<10 hours' in once}")


def run_step_2_retrieval(
    query: Optional[str] = None,
    source: str = "all",
    asin: Optional[str] = None,
    top_k: int = 3,
) -> None:
    """Bước 2: Truy xuất BM25 trực tiếp trên CPU cho AmazonQA và Shopify Legal."""
    _print_header("BUOC 2: TRUY XUAT TRUC TIEP TREN CPU (BM25)")
    chunks_path = ROOT / "data" / "processed" / "chunks_dev_recursive.jsonl"
    policy_path = ROOT / "data" / "processed" / "policy_docs_shopify.jsonl"

    if source in ("all", "amazon"):
        chunks = load_jsonl_records(chunks_path)
        if not chunks:
            print(f"- Khong tim thay {chunks_path.relative_to(ROOT)}. Chay 'python -m src.run chunk_strategies' truoc.")
        else:
            q_amz = query or "110 or 220 volt power adapter input voltage"
            target_asin = asin if query else (asin or "B002RYYZZS")
            hits, ms = search_bm25_chunks(chunks, q_amz, top_k=top_k, asin=target_asin)
            scope = f"ASIN={target_asin}" if target_asin else "Toan kho Global"
            print(f"- Amazon Electronics ({scope}) | Truy van: \"{q_amz}\" | Thoi gian: {ms:.2f} ms")
            for rank, (chunk, score) in enumerate(hits, start=1):
                print(
                    f"  #{rank} [{chunk.get('chunk_id')}] score={score:.3f} | "
                    f"{_shorten(chunk.get('text', ''), 160)}"
                )

    if source in ("all", "shopify"):
        passages = load_shopify_passages(policy_path)
        if not passages:
            print(f"- Khong tim thay {policy_path.relative_to(ROOT)}. Chay 'python -m src.run build_policy_corpus' truoc.")
        else:
            q_shop = query or "What financial remedies does Shopify provide for chargebacks?"
            hits, ms = search_bm25_chunks(passages, q_shop, top_k=top_k)
            print(f"- Shopify Legal (527 passages) | Truy van: \"{q_shop}\" | Thoi gian: {ms:.2f} ms")
            for rank, (passage, score) in enumerate(hits, start=1):
                print(
                    f"  #{rank} [{passage.get('passage_id')}] score={score:.3f} | "
                    f"{_shorten(passage.get('text', ''), 160)}"
                )


def run_step_3_rag_comparison() -> None:
    """Bước 3: Đối chiếu câu trả lời thật giữa No-RAG và RAG (Hybrid RRF)."""
    _print_header("BUOC 3: DOI CHIEU NO-RAG VS RAG (RQ2 & SHOPIFY)")
    eval_dev = load_jsonl_index(ROOT / "data" / "eval" / "eval_120_dev_labeled.jsonl", "qid_moi")
    no_rag = load_jsonl_index(ROOT / "data" / "processed" / "eval_judge_no_rag_dev.jsonl", "qid_moi")
    rag_hyb = load_jsonl_index(ROOT / "data" / "processed" / "eval_judge_rag_hybrid_recursive_dev_ollama.jsonl", "qid_moi")
    eval_pol = load_jsonl_index(ROOT / "data" / "eval" / "eval_policy_shopify.jsonl", "qid_moi")
    judge_pol = load_jsonl_index(ROOT / "data" / "processed" / "eval_judge_policy_shopify_ollama.jsonl", "qid_moi")

    if not (eval_dev and no_rag and rag_hyb):
        print("- Thieu du lieu ket qua dev trong data/eval hoac data/processed.")
        return

    for qid, label in [
        ("q-dev-0253", "Cau du bang chung (yes)"),
        ("q-dev-0589", "Cau mo rong toan tai lieu (partial)"),
        ("q-dev-0888", "Cau bay khong co dap an (no)"),
    ]:
        q_info = eval_dev.get(qid, {})
        r0 = no_rag.get(qid, {})
        r1 = rag_hyb.get(qid, {})
        print(f"- {qid} [{label}] ({q_info.get('asin', '')}): \"{_shorten(q_info.get('question_text_clean', ''), 110)}\"")
        print(
            f"  No-RAG : Faith={r0.get('faithfulness')} Hallu={r0.get('has_hallucination')} | "
            f"{_shorten(r0.get('generated_answer', ''), 150)}"
        )
        print(
            f"  RAG Hyb: Faith={r1.get('faithfulness')} Hallu={r1.get('has_hallucination')} | "
            f"{_shorten(r1.get('generated_answer', ''), 150)}"
        )

    if eval_pol and judge_pol and "pol-m-01" in judge_pol:
        pq = eval_pol.get("pol-m-01", {})
        pj = judge_pol["pol-m-01"]
        print(f"- pol-m-01 [Shopify multi_condition]: \"{_shorten(pq.get('question_text_clean', ''), 110)}\"")
        print(
            f"  Policy RAG: CP@5={pj.get('context_precision'):.4f} Faith={pj.get('faithfulness')} "
            f"Rel={pj.get('answer_relevance')} | {_shorten(pj.get('generated_answer', ''), 150)}"
        )


def run_step_4_counterfactual() -> None:
    """Bước 4: Minh họa thí nghiệm phản thực tế (chống rò rỉ dữ liệu 2019)."""
    _print_header("BUOC 4: THI NGHIEM PHAN THUC TE (COUNTERFACTUAL N=30)")
    cf_eval = load_jsonl_index(ROOT / "data" / "eval" / "eval_counterfactual_30_dev.jsonl", "qid_moi")
    cf_single = load_jsonl_records(ROOT / "data" / "processed" / "eval_judge_counterfactual_30_dev_ollama.jsonl")
    cf_noisy = load_jsonl_records(ROOT / "data" / "processed" / "eval_judge_counterfactual_30_noisy_top5.jsonl")

    if not cf_single:
        print("- Thieu ket qua phan thuc te trong data/processed/.")
        return

    adhere = sum(int(r.get("adheres_counterfactual", 0)) for r in cf_single)
    revert = sum(int(r.get("reverts_to_prior", 0)) for r in cf_single)
    print(f"- Thiet lap don doan: Tuan thu CF = {adhere}/{len(cf_single)} ({adhere/len(cf_single)*100:.2f}%) | Ro ri ky uc cu = {revert}/{len(cf_single)}")

    if cf_noisy:
        strict = sum(int(r.get("strict_cf_only", 0)) for r in cf_noisy)
        hedged = sum(int(r.get("hedged_conflict", 0)) for r in cf_noisy)
        print(f"- Thiet lap nhieu Top-5: Tuan thu CF = {strict}/{len(cf_noisy)} ({strict/len(cf_noisy)*100:.2f}%) | Bao cao mau thuan = {hedged}/{len(cf_noisy)}")

    sample_q = cf_eval.get("q-dev-0253", {})
    sample_r = next((r for r in cf_single if r.get("qid_moi") == "q-dev-0253"), {})
    if sample_q and sample_r:
        print(f"- Ca mau q-dev-0253: Su that 2019 = {_shorten(sample_q.get('original_fact', ''), 80)}")
        print(f"  Gia lap CF  = {_shorten(sample_q.get('counterfactual_fact', ''), 90)}")
        print(f"  Tra loi RAG = {_shorten(sample_r.get('generated_answer', ''), 150)}")


def run_step_5_test_summary() -> None:
    """Bước 5: Tổng hợp kết quả chốt một lần duy nhất trên tập test độc lập."""
    _print_header("BUOC 5: KET QUA CHOT TAP TEST DOC LAP (N=120)")
    configs = [
        ("no_rag (Doi chung)", "eval_judge_test_no_rag_ollama.jsonl"),
        ("fixed_vector (Baseline)", "eval_judge_test_fixed_vector_ollama.jsonl"),
        ("recursive_hybrid (Toi uu)", "eval_judge_test_recursive_hybrid_ollama.jsonl"),
        ("parent_child_hybrid", "eval_judge_test_parent_child_hybrid_ollama.jsonl"),
    ]
    print(f"  {'Cau hinh':<28} {'Direct':>8} {'CP@5':>8} {'Faith(Dir)':>11} {'Ao giac(Dir)':>14}")
    for label, filename in configs:
        recs = load_jsonl_records(ROOT / "data" / "processed" / filename)
        if not recs:
            print(f"  {label:<28} [Chua co tep {filename}]")
            continue
        s = summarize_test_records(recs)
        hallu_str = f"{s['hallucination_rate_pct']:5.2f}% ({s['hallucinations']}/{s['direct']})"
        print(
            f"  {label:<28} {s['direct']:>3}/{s['total']:<4} "
            f"{s['cp5_answerable']:>8.4f} {s['faithfulness_direct']:>11.4f} {hallu_str:>14}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Khởi tạo bộ phân tích tham số dòng lệnh cho CLI demo."""
    parser = argparse.ArgumentParser(description="Demo CLI he thong hoi dap RAG thuong mai dien tu")
    parser.add_argument(
        "--step",
        choices=["all", "1", "2", "3", "4", "5"],
        default="all",
        help="Chon buoc demo (1: lam sach, 2: truy xuat BM25, 3: No-RAG vs RAG, 4: Counterfactual, 5: ket qua test)",
    )
    parser.add_argument("--query", type=str, default=None, help="Cau hoi tu nhap de truy xuat BM25 truc tiep")
    parser.add_argument(
        "--source",
        choices=["all", "amazon", "shopify"],
        default="all",
        help="Nguon du lieu truy xuat khi dung --query",
    )
    parser.add_argument("--asin", type=str, default=None, help="Loc ma san pham ASIN (chi ap dung cho nguon amazon)")
    parser.add_argument("--top-k", type=int, default=3, help="So doan truy xuat Top-K hien thi")
    parser.add_argument("--live", action="store_true", help="Bat che do hoi dap truc tiep (truy xuat + goi LLM tra loi)")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b", help="Ten mo hinh sinh khi chay --live")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Điểm vào chính của giao diện demo CLI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = build_parser().parse_args(argv)
    if args.live:
        live_source = "amazon" if args.source == "all" else args.source
        live_k = args.top_k if args.top_k != 3 else 5
        if args.query:
            run_live_qa(
                query=args.query,
                source=live_source,
                asin=args.asin,
                top_k=live_k,
                model_name=args.model,
            )
            return 0
        print(f"Che do hoi dap truc tiep (source={live_source}, asin={args.asin or 'Global'}). Go 'exit' de thoat.")
        while True:
            try:
                user_q = input("\nNhap cau hoi: ").strip()
            except EOFError:
                break
            if not user_q or user_q.lower() in ("exit", "quit", "q"):
                break
            run_live_qa(
                query=user_q,
                source=live_source,
                asin=args.asin,
                top_k=live_k,
                model_name=args.model,
            )
        return 0

    if args.query:
        run_step_2_retrieval(query=args.query, source=args.source, asin=args.asin, top_k=args.top_k)
        return 0

    steps = {
        "1": run_step_1_clean,
        "2": lambda: run_step_2_retrieval(source=args.source, asin=args.asin, top_k=args.top_k),
        "3": run_step_3_rag_comparison,
        "4": run_step_4_counterfactual,
        "5": run_step_5_test_summary,
    }
    if args.step == "all":
        for key in ("1", "2", "3", "4", "5"):
            steps[key]()
    else:
        steps[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
