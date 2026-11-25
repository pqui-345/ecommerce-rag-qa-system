"""Điểm thực thi chính (CLI entry point) cho các lệnh trong dự án."""

import argparse
import sys
from pathlib import Path
import yaml

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.merge import gop_du_lieu
from src.working_set import bo_lam_viec
from src.clean import ap_dung_lam_sach, ap_dung_lam_sach_questions
from src.label_verification import tong_hop_kiem_dinh_nhan
from src.chunking import STRATEGIES_NANG_CAO, chay_chunk_baseline, chay_chunk_strategies
from src.retrieval_bm25 import chay_danh_gia_bm25_baseline
from src.retrieval_vector import BGEFastEmbedder, chay_danh_gia_vector_baseline
from src.generation_baseline import (
    chay_generation_no_rag_baseline,
    chay_generation_rag_baseline,
)
from src.evaluation_judge import chay_danh_gia_judge_baseline
from src.chunking_eval import ALL_STRATEGIES, chay_danh_gia_truy_xuat_6_chien_luoc
from src.chunking_rag_eval import chay_rag_va_judge_cac_chien_luoc
from src.hybrid_retrieval import chay_hybrid_6_chien_luoc
from src.policy_corpus import xay_dung_kho_chinh_sach
from src.specialized_eval import (
    chay_danh_gia_counterfactual,
    chay_rag_va_judge_chinh_sach,
    danh_gia_truy_xuat_chinh_sach,
    xay_dung_bo_cau_hoi_chinh_sach,
    xay_dung_bo_counterfactual_30,
)
from src.test_set_eval import (
    chay_danh_gia_chot_tap_test,
    chuan_bi_truy_xuat_test,
    lay_mau_phan_tang_test_120,
)


def load_config(config_path: Path) -> dict:
    """Đọc tệp cấu hình config.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / "config.yaml"
    config = load_config(config_file)

    default_val = project_root / config.get("val_path", "data/raw/val-qar.jsonl")
    default_test = project_root / config.get("test_path", "data/raw/test-qar.jsonl")
    default_out = project_root / config.get("out_path_v2", "data/merged/amazonqa_merged_v2.jsonl")
    default_report = project_root / config.get("report_path", "reports/gop_v2.md")
    dedup_key = config.get("dedup_key", "qid")
    log_file = project_root / "logs" / "gop_du_lieu.log"

    default_merged = project_root / config.get(
        "out_path_v2", "data/merged/amazonqa_merged_v2.jsonl"
    )
    default_processed_dir = project_root / "data" / "processed"
    default_report_dir = project_root / "reports"
    default_bo_log = project_root / "logs" / "bo_lam_viec.log"
    default_category = config.get("category", "Electronics")
    default_min_doc_chars = config.get("min_doc_chars", 5000)
    default_n_asin_dev = config.get("n_asin_dev", 150)
    default_n_asin_test = config.get("n_asin_test", 150)
    default_seed = config.get("seed", 42)
    default_min_question_chars = config.get("min_question_chars", 10)
    default_n_kiem_tay = config.get("n_kiem_tay", 120)

    parser = argparse.ArgumentParser(description="Chương trình thực thi RAG thương mại điện tử.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gop_parser = subparsers.add_parser("gop", help="Gộp hai file dữ liệu val và test.")
    gop_parser.add_argument(
        "--val",
        type=Path,
        default=default_val,
        help="Đường dẫn tới file val-qar.jsonl (mặc định lấy từ config.yaml)",
    )
    gop_parser.add_argument(
        "--test",
        type=Path,
        default=default_test,
        help="Đường dẫn tới file test-qar.jsonl (mặc định lấy từ config.yaml)",
    )
    gop_parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Đường dẫn tới file đầu ra amazonqa_merged_v2.jsonl (mặc định lấy từ config.yaml)",
    )
    gop_parser.add_argument(
        "--report",
        type=Path,
        default=default_report,
        help="Đường dẫn file báo cáo kết quả gộp (mặc định lấy từ config.yaml)",
    )

    blv_parser = subparsers.add_parser(
        "bo_lam_viec", help="Dựng bộ làm việc nhỏ từ amazonqa_merged_v2.jsonl."
    )
    blv_parser.add_argument(
        "--merged",
        type=Path,
        default=default_merged,
        help="Đường dẫn file merged v2 (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục đầu ra data/processed (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--report-dir",
        type=Path,
        default=default_report_dir,
        help="Thư mục báo cáo reports (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--category",
        type=str,
        default=default_category,
        help="Danh mục lọc (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=default_min_doc_chars,
        help="Số ký tự tối thiểu của tài liệu (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--n-asin-dev",
        type=int,
        default=default_n_asin_dev,
        help="Số asin chọn cho dev (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--n-asin-test",
        type=int,
        default=default_n_asin_test,
        help="Số asin chọn cho test (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="Seed ngẫu nhiên (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--min-question-chars",
        type=int,
        default=default_min_question_chars,
        help="Độ dài questionText tối thiểu (mặc định lấy từ config.yaml)",
    )
    blv_parser.add_argument(
        "--n-kiem-tay",
        type=int,
        default=default_n_kiem_tay,
        help="Số câu kiểm tay (mặc định lấy từ config.yaml)",
    )

    ls_parser = subparsers.add_parser(
        "lam_sach", help="Làm sạch văn bản docs_dev/docs_test -> docs_*_clean.jsonl."
    )
    ls_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục chứa docs_*.jsonl và nơi ghi *_clean.jsonl",
    )
    ls_parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=default_min_doc_chars,
        help="Ngưỡng đếm tài liệu dưới ngưỡng sau làm sạch (từ config.yaml)",
    )
    ls_parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="Seed để chọn 3 ví dụ trước/sau (từ config.yaml)",
    )

    lsq_parser = subparsers.add_parser(
        "lam_sach_questions",
        help="Làm sạch questions_dev/questions_test -> questions_*_clean.jsonl.",
    )
    lsq_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục chứa questions_*.jsonl và nơi ghi questions_*_clean.jsonl",
    )

    kdn_parser = subparsers.add_parser(
        "kiem_dinh_nhan",
        help="Tính Cohen's kappa cho nhãn kiểm tay 120 câu và xuất data/eval/eval_120_dev_labeled.jsonl.",
    )
    kdn_parser.add_argument(
        "--ung-vien-csv",
        type=Path,
        default=default_report_dir / "ung_vien_kiem_tay.csv",
        help="Đường dẫn file ung_vien_kiem_tay.csv",
    )
    kdn_parser.add_argument(
        "--ra-soat-csv",
        type=Path,
        default=default_report_dir / "kiem_tra_93_cau_da_ra_soat_chot.csv",
        help="Đường dẫn file kiem_tra_93_cau_da_ra_soat_chot.csv",
    )
    kdn_parser.add_argument(
        "--questions-clean",
        type=Path,
        default=default_processed_dir / "questions_dev_clean.jsonl",
        help="Đường dẫn file questions_dev_clean.jsonl",
    )
    kdn_parser.add_argument(
        "--eval-out",
        type=Path,
        default=project_root / "data" / "eval" / "eval_120_dev_labeled.jsonl",
        help="Đường dẫn file đầu ra trong data/eval/",
    )
    kdn_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "ket_qua_kiem_dinh_nhan.md",
        help="Đường dẫn báo cáo Markdown đầu ra",
    )

    default_chunk_size_words = config.get("chunk_size_words", 200)
    default_chunk_overlap_words = config.get("chunk_overlap_words", 40)

    cb_parser = subparsers.add_parser(
        "chunk_baseline",
        help="Phân đoạn Fixed-size cơ sở cho docs_dev_clean và docs_test_clean.",
    )
    cb_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục chứa docs_*_clean.jsonl và nơi ghi chunks_*_fixed.jsonl",
    )
    cb_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "chunk_baseline_fixed.md",
        help="Đường dẫn báo cáo Markdown kết quả phân đoạn cơ sở",
    )
    cb_parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=default_chunk_size_words,
        help="Số từ tối đa mỗi chunk (mặc định từ config.yaml)",
    )
    cb_parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        default=default_chunk_overlap_words,
        help="Số từ gối đầu giữa 2 chunk liên tiếp (mặc định từ config.yaml)",
    )

    default_top_k = config.get("top_k", 5)

    rbm_parser = subparsers.add_parser(
        "retrieve_bm25",
        help="Truy xuất cơ sở bằng BM25 trên chunks_dev_fixed.jsonl cho 120 câu hỏi đánh giá.",
    )
    rbm_parser.add_argument(
        "--chunks",
        type=Path,
        default=default_processed_dir / "chunks_dev_fixed.jsonl",
        help="Đường dẫn file chunks đầu vào",
    )
    rbm_parser.add_argument(
        "--eval-questions",
        type=Path,
        default=project_root / "data" / "eval" / "eval_120_dev_labeled.jsonl",
        help="Đường dẫn file 120 câu hỏi đánh giá",
    )
    rbm_parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=default_processed_dir / "retrieval_bm25_dev_fixed.jsonl",
        help="Đường dẫn file JSONL kết quả truy xuất đầu ra",
    )
    rbm_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "retrieval_bm25_baseline.md",
        help="Đường dẫn file báo cáo Markdown đầu ra",
    )
    rbm_parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help="Số đoạn truy xuất top-k (mặc định từ config.yaml)",
    )

    default_embedding_model = config.get("embedding_model", "BAAI/bge-small-en-v1.5")

    rvec_parser = subparsers.add_parser(
        "retrieve_vector",
        help="Truy xuất cơ sở bằng Dense Vector (BAAI/bge-small-en-v1.5 + ChromaDB) cho 120 câu hỏi.",
    )
    rvec_parser.add_argument(
        "--chunks",
        type=Path,
        default=default_processed_dir / "chunks_dev_fixed.jsonl",
        help="Đường dẫn file chunks đầu vào",
    )
    rvec_parser.add_argument(
        "--eval-questions",
        type=Path,
        default=project_root / "data" / "eval" / "eval_120_dev_labeled.jsonl",
        help="Đường dẫn file 120 câu hỏi đánh giá",
    )
    rvec_parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=default_processed_dir / "retrieval_vector_dev_fixed.jsonl",
        help="Đường dẫn file JSONL kết quả truy xuất vector đầu ra",
    )
    rvec_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "retrieval_vector_baseline.md",
        help="Đường dẫn file báo cáo Markdown đầu ra",
    )
    rvec_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=project_root / "data" / "vectordb" / "chroma_dev_fixed",
        help="Thư mục lưu trữ ChromaDB",
    )
    rvec_parser.add_argument(
        "--model-name",
        type=str,
        default=default_embedding_model,
        help="Tên mô hình embedding (mặc định BAAI/bge-small-en-v1.5)",
    )
    rvec_parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help="Số đoạn truy xuất top-k (mặc định từ config.yaml)",
    )

    default_generator_model = config.get("generator_model", "qwen/qwen3.8-27b")

    gnorag_parser = subparsers.add_parser(
        "generate_no_rag",
        help="Sinh câu trả lời đối chứng không ngữ cảnh (Baseline 0: No-RAG) cho 120 câu hỏi.",
    )
    gnorag_parser.add_argument(
        "--eval-questions",
        type=Path,
        default=project_root / "data" / "eval" / "eval_120_dev_labeled.jsonl",
        help="Đường dẫn file 120 câu hỏi đánh giá",
    )
    gnorag_parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=default_processed_dir / "generations_no_rag_dev.jsonl",
        help="Đường dẫn file JSONL kết quả sinh câu trả lời No-RAG",
    )
    gnorag_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "generation_no_rag_baseline.md",
        help="Đường dẫn file báo cáo Markdown đầu ra",
    )
    gnorag_parser.add_argument(
        "--model-name",
        type=str,
        default=default_generator_model,
        help="Tên mô hình LLM trên Groq (mặc định qwen/qwen3.8-27b)",
    )
    gnorag_parser.add_argument(
        "--delay-sec",
        type=float,
        default=0.8,
        help="Khoảng nghỉ giữa 2 request để tránh vượt TPM",
    )

    grag_parser = subparsers.add_parser(
        "generate_rag",
        help="Sinh câu trả lời Naive RAG cơ sở (Fixed-size + BGE-small-en-v1.5 Top-5) cho 120 câu hỏi.",
    )
    grag_parser.add_argument(
        "--retrieval-jsonl",
        type=Path,
        default=default_processed_dir / "retrieval_vector_dev_fixed.jsonl",
        help="Đường dẫn file kết quả truy xuất đầu vào",
    )
    grag_parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=default_processed_dir / "generations_rag_fixed_dev.jsonl",
        help="Đường dẫn file JSONL kết quả sinh câu trả lời Naive RAG",
    )
    grag_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "generation_rag_fixed_baseline.md",
        help="Đường dẫn file báo cáo Markdown đầu ra",
    )
    grag_parser.add_argument(
        "--model-name",
        type=str,
        default=default_generator_model,
        help="Tên mô hình LLM trên Groq (mặc định qwen/qwen3.8-27b)",
    )
    grag_parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help="Số đoạn ngữ cảnh đưa vào prompt (mặc định 5)",
    )
    grag_parser.add_argument(
        "--delay-sec",
        type=float,
        default=1.2,
        help="Khoảng nghỉ cơ sở giữa 2 request",
    )

    default_judge_model = config.get("judge_model", "gemini-3.5-flash")

    ej_parser = subparsers.add_parser(
        "evaluate_judge",
        help="Chấm điểm RAGAS & Hallucination bằng Giám khảo Gemini và tính Cohen's kappa (RQ2 & RQ3).",
    )
    ej_parser.add_argument(
        "--eval-labeled",
        type=Path,
        default=project_root / "data" / "eval" / "eval_120_dev_labeled.jsonl",
        help="Đường dẫn file 120 câu hỏi có nhãn người",
    )
    ej_parser.add_argument(
        "--no-rag-gen",
        type=Path,
        default=default_processed_dir / "generations_no_rag_dev.jsonl",
        help="Đường dẫn file câu trả lời Baseline 0 (No-RAG)",
    )
    ej_parser.add_argument(
        "--rag-gen",
        type=Path,
        default=default_processed_dir / "generations_rag_fixed_dev.jsonl",
        help="Đường dẫn file câu trả lời Baseline 1 (Naive RAG)",
    )
    ej_parser.add_argument(
        "--out-no-rag",
        type=Path,
        default=default_processed_dir / "eval_judge_no_rag_dev.jsonl",
        help="Đường dẫn file JSONL kết quả chấm No-RAG",
    )
    ej_parser.add_argument(
        "--out-rag",
        type=Path,
        default=default_processed_dir / "eval_judge_rag_fixed_dev.jsonl",
        help="Đường dẫn file JSONL kết quả chấm Naive RAG",
    )
    ej_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "evaluation_baseline_ragas_judge.md",
        help="Đường dẫn file báo cáo Markdown kết quả đánh giá Baseline",
    )
    ej_parser.add_argument(
        "--judge-model",
        type=str,
        default=default_judge_model,
        help="Tên mô hình Giám khảo Gemini (mặc định gemini-3.5-flash)",
    )
    ej_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Số câu hỏi chấm trong mỗi batch gọi Gemini",
    )

    cs_parser = subparsers.add_parser(
        "chunk_strategies",
        help="Sinh chunks cho 5 chiến lược phân đoạn nâng cao trên tập dev.",
    )
    cs_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục data/processed",
    )
    cs_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "chunking_strategies_stats.md",
        help="Đường dẫn báo cáo thống kê các chiến lược phân đoạn",
    )
    cs_parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES_NANG_CAO),
        help="Danh sách chiến lược cần chạy",
    )

    rs_parser = subparsers.add_parser(
        "retrieve_strategies",
        help="Đánh giá truy xuất BM25 và Dense Vector cho toàn bộ 6 chiến lược phân đoạn (RQ1).",
    )
    rs_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục data/processed",
    )
    rs_parser.add_argument(
        "--eval-questions",
        type=Path,
        default=project_root / "data" / "eval" / "eval_120_dev_labeled.jsonl",
        help="Đường dẫn 120 câu hỏi đánh giá",
    )
    rs_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "chunking_retrieval_comparison.md",
        help="Đường dẫn báo cáo đối sánh truy xuất 6 chiến lược",
    )
    rs_parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(ALL_STRATEGIES),
        help="Danh sách chiến lược cần đánh giá",
    )

    rags_parser = subparsers.add_parser(
        "rag_strategies",
        help="Chạy Sinh câu trả lời (Groq) + Giám khảo RAGAS (Gemini) cho các chiến lược phân đoạn.",
    )
    rags_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục data/processed",
    )
    rags_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "chunking_generation_ragas_comparison.md",
        help="Đường dẫn báo cáo đối sánh RAGAS giữa các chiến lược phân đoạn",
    )
    rags_parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(ALL_STRATEGIES),
        help="Danh sách chiến lược cần chạy RAG + Judge",
    )
    rags_parser.add_argument(
        "--generator-model",
        type=str,
        default=default_generator_model,
        help="Tên mô hình sinh câu trả lời (hoặc tên model trên Ollama/vLLM Colab)",
    )
    rags_parser.add_argument(
        "--gen-delay",
        type=float,
        default=0.4,
        help="Độ trễ giữa các câu gọi LLM sinh (đặt 0 khi chạy trên Colab/local)",
    )
    rags_parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Chỉ sinh câu trả lời, bỏ qua bước chấm Gemini Judge",
    )
    rags_parser.add_argument(
        "--retrieval",
        type=str,
        choices=("vector", "hybrid"),
        default="vector",
        help="Nhánh truy xuất đầu vào: vector hoặc hybrid",
    )
    rags_parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Hậu tố tên file đầu ra (ví dụ _ollama để không ghi đè file cũ)",
    )

    rh_parser = subparsers.add_parser(
        "retrieve_hybrid",
        help="Chạy truy xuất lai (BM25 + Dense Vector bằng RRF) trên các chiến lược phân đoạn.",
    )
    rh_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed_dir,
        help="Thư mục data/processed",
    )
    rh_parser.add_argument(
        "--eval-questions",
        type=Path,
        default=project_root / "data" / "eval" / "eval_120_dev_labeled.jsonl",
        help="Đường dẫn 120 câu hỏi đánh giá",
    )
    rh_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "hybrid_retrieval_comparison.md",
        help="Đường dẫn báo cáo đối sánh truy xuất lai",
    )
    rh_parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(ALL_STRATEGIES),
        help="Danh sách chiến lược cần chạy truy xuất lai",
    )
    rh_parser.add_argument(
        "--k0",
        type=int,
        default=60,
        help="Hằng số làm mịn RRF k0 (mặc định 60)",
    )
    rh_parser.add_argument(
        "--depth",
        type=int,
        default=50,
        help="Độ sâu bảng xếp hạng mỗi nhánh trước khi hợp nhất RRF (mặc định 50)",
    )

    pol_parser = subparsers.add_parser(
        "build_policy_corpus",
        help="Thu thập và chuẩn hóa kho tài liệu chính sách TMĐT công khai (Shopify Legal).",
    )
    pol_parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=default_processed_dir / "policy_docs_shopify.jsonl",
        help="Đường dẫn file JSONL kho tài liệu chính sách đầu ra",
    )
    pol_parser.add_argument(
        "--report-out",
        type=Path,
        default=default_report_dir / "policy_corpus_stats.md",
        help="Đường dẫn báo cáo thống kê kho tài liệu chính sách",
    )

    sp_parser = subparsers.add_parser(
        "build_specialized_eval",
        help="Tạo bộ 60 câu chính sách Shopify + 30 câu Counterfactual và đo Recall@k/MRR@5 trên kho chính sách.",
    )
    sp_parser.add_argument("--processed-dir", type=Path, default=default_processed_dir)
    sp_parser.add_argument("--eval-dir", type=Path, default=project_root / "data" / "eval")
    sp_parser.add_argument("--report-dir", type=Path, default=default_report_dir)

    epr_parser = subparsers.add_parser(
        "eval_policy_rag",
        help="Sinh câu trả lời và chấm RAGAS trên 60 câu hỏi chính sách Shopify.",
    )
    epr_parser.add_argument("--processed-dir", type=Path, default=default_processed_dir)
    epr_parser.add_argument("--report-out", type=Path, default=default_report_dir / "policy_rag_evaluation_ollama.md")
    epr_parser.add_argument("--generator-model", type=str, default=default_generator_model)
    epr_parser.add_argument("--gen-delay", type=float, default=0.0)

    ecf_parser = subparsers.add_parser(
        "eval_counterfactual",
        help="Chạy thí nghiệm Phản thực tế (30 câu Counterfactual) và chấm Counterfactual Adherence Rate.",
    )
    ecf_parser.add_argument("--processed-dir", type=Path, default=default_processed_dir)
    ecf_parser.add_argument("--eval-dir", type=Path, default=project_root / "data" / "eval")
    ecf_parser.add_argument("--report-out", type=Path, default=default_report_dir / "counterfactual_evaluation_ollama.md")
    ecf_parser.add_argument("--generator-model", type=str, default=default_generator_model)
    ecf_parser.add_argument("--gen-delay", type=float, default=0.0)

    pte_parser = subparsers.add_parser(
        "prepare_test_eval",
        help="Lấy mẫu 120 câu tập test độc lập và chạy truy xuất 3 cấu hình đã khóa từ dev.",
    )
    pte_parser.add_argument("--processed-dir", type=Path, default=default_processed_dir)
    pte_parser.add_argument("--eval-dir", type=Path, default=project_root / "data" / "eval")
    pte_parser.add_argument("--report-out", type=Path, default=default_report_dir / "test_retrieval_final.md")

    rtj_parser = subparsers.add_parser(
        "run_test_rag_judge",
        help="Chạy đánh giá chốt 1 lần duy nhất (Generator + Gemini Judge) trên 120 câu tập test.",
    )
    rtj_parser.add_argument("--processed-dir", type=Path, default=default_processed_dir)
    rtj_parser.add_argument("--eval-dir", type=Path, default=project_root / "data" / "eval")
    rtj_parser.add_argument("--report-out", type=Path, default=default_report_dir / "test_evaluation_final_ollama.md")
    rtj_parser.add_argument("--generator-model", type=str, default=default_generator_model)
    rtj_parser.add_argument("--gen-delay", type=float, default=0.0)

    args = parser.parse_args()

    if args.command == "gop":
        val_path = args.val if args.val.is_absolute() else (project_root / args.val)
        test_path = args.test if args.test.is_absolute() else (project_root / args.test)
        out_path = args.out if args.out.is_absolute() else (project_root / args.out)
        report_path = args.report if args.report.is_absolute() else (project_root / args.report)

        gop_du_lieu(
            val_path=val_path,
            test_path=test_path,
            out_path=out_path,
            dedup_key=dedup_key,
            log_path=log_file,
            report_path=report_path,
        )

    elif args.command == "bo_lam_viec":
        merged_path = args.merged if args.merged.is_absolute() else (project_root / args.merged)
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else (project_root / args.processed_dir)
        report_dir = args.report_dir if args.report_dir.is_absolute() else (project_root / args.report_dir)
        log_path = default_bo_log

        bo_lam_viec(
            merged_path=merged_path,
            processed_dir=processed_dir,
            report_dir=report_dir,
            log_path=log_path,
            category=args.category,
            min_doc_chars=args.min_doc_chars,
            n_asin_dev=args.n_asin_dev,
            n_asin_test=args.n_asin_test,
            seed=args.seed,
            min_question_chars=args.min_question_chars,
            n_kiem_tay=args.n_kiem_tay,
        )

    elif args.command == "lam_sach":
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else (project_root / args.processed_dir)
        ket_qua = ap_dung_lam_sach(
            processed_dir=processed_dir,
            min_doc_chars=args.min_doc_chars,
            seed=args.seed,
        )
        for ten in ("dev", "test"):
            r = ket_qua["ket_qua"][ten]
            print(f"===== {ten} =====")
            print(
                f"so_dong_vao={r['n_in']} so_dong_ra={r['n_out']} "
                f"doc_id_khong_doi_va_duy_nhat={'DAT' if r['kiem_doc_id'] else 'KHONG DAT'} "
                f"so_dong_khop={'DAT' if r['kiem_so_dong'] else 'KHONG DAT'} "
                f"doc_rong_sau_sach={r['n_empty_after']} "
                f"ton_du_entity_thap_phan={r['n_residual_num']} "
                f"ton_du_entity_ten={r['n_residual_named']} "
                f"doc_duoi_{args.min_doc_chars}={r['n_below_min_5000']}"
            )

    elif args.command == "lam_sach_questions":
        processed_dir = (
            args.processed_dir
            if args.processed_dir.is_absolute()
            else (project_root / args.processed_dir)
        )
        ket_qua = ap_dung_lam_sach_questions(processed_dir=processed_dir)
        for ten in ("dev", "test"):
            r = ket_qua[ten]
            print(f"===== questions_{ten}_clean =====")
            print(
                f"so_dong_vao={r['n_in']} so_dong_ra={r['n_out']} "
                f"dong_hong={r['n_corrupt']} "
                f"qid_duy_nhat={'DAT' if r['kiem_qid_duy_nhat'] else 'KHONG DAT'} "
                f"so_dong_khop={'DAT' if r['kiem_so_dong'] else 'KHONG DAT'} "
                f"q_thay_doi={r['n_q_changed']} a_thay_doi={r['n_a_changed']} "
                f"snippet_thay_doi={r['n_s_changed']} "
                f"q_rong_sau_sach={r['n_empty_q_after']} "
                f"ref_rong_sau_sach={r['n_empty_ref_after']}"
            )

    elif args.command == "kiem_dinh_nhan":
        ket_qua = tong_hop_kiem_dinh_nhan(
            ung_vien_csv=args.ung_vien_csv,
            ra_soat_93_csv=args.ra_soat_csv,
            questions_clean_jsonl=args.questions_clean,
            eval_out_jsonl=args.eval_out,
            report_out_md=args.report_out,
        )
        print("===== KIEM DINH NHAN 120 CAU DEV =====")
        print(f"so_cau_eval={ket_qua['n_eval']}")
        print(f"phan_bo_3_muc={ket_qua['dem_3_muc']}")
        print(f"phan_bo_chi_tiet={ket_qua['dem_chi_tiet']}")
        print(f"kappa_strict_10={ket_qua['metrics_strict_10']['kappa']}")
        print(f"kappa_lenient_10={ket_qua['metrics_lenient_10']['kappa']}")
        print(f"kappa_full_doc={ket_qua['metrics_full_doc']['kappa']}")
        print(f"eval_out={ket_qua['eval_out_jsonl']}")
        print(f"report_out={ket_qua['report_out_md']}")

    elif args.command == "chunk_baseline":
        processed_dir = (
            args.processed_dir
            if args.processed_dir.is_absolute()
            else (project_root / args.processed_dir)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        ket_qua = chay_chunk_baseline(
            processed_dir=processed_dir,
            report_path=report_out,
            chunk_size_words=args.chunk_size_words,
            chunk_overlap_words=args.chunk_overlap_words,
        )
        for ten in ("dev", "test"):
            r = ket_qua[ten]
            print(f"===== chunks_{ten}_fixed =====")
            print(
                f"so_docs={r['n_docs_in']} so_chunks={r['n_chunks_out']} "
                f"dong_hong={r['n_corrupt']} doc_rong={r['n_empty_docs']} "
                f"chunk_id_duy_nhat={'DAT' if r['kiem_chunk_id_duy_nhat'] else 'KHONG DAT'} "
                f"chunks_moi_doc_trung_vi={r['stats_chunks_per_doc']['median']} "
                f"words_moi_chunk_trung_vi={r['stats_words_per_chunk']['median']} "
                f"chars_moi_chunk_trung_vi={r['stats_chars_per_chunk']['median']}"
            )
        print(f"report_out={ket_qua['report_path']}")

    elif args.command == "retrieve_bm25":
        chunks_path = (
            args.chunks
            if args.chunks.is_absolute()
            else (project_root / args.chunks)
        )
        eval_q_path = (
            args.eval_questions
            if args.eval_questions.is_absolute()
            else (project_root / args.eval_questions)
        )
        out_jsonl = (
            args.out_jsonl
            if args.out_jsonl.is_absolute()
            else (project_root / args.out_jsonl)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        kq = chay_danh_gia_bm25_baseline(
            chunks_path=chunks_path,
            eval_questions_path=eval_q_path,
            out_jsonl_path=out_jsonl,
            report_path=report_out,
            top_k=args.top_k,
        )
        print("===== RETRIEVAL BM25 BASELINE =====")
        print(
            f"n_chunks={kq['n_chunks']} n_questions={kq['n_questions']} "
            f"n_questions_with_gt={kq['n_questions_with_gt']} "
            f"corrupt_chunks={kq['n_corrupt_chunks']} corrupt_q={kq['n_corrupt_questions']}"
        )
        print(f"global_asin={kq['global_asin']}")
        print(f"global_evidence={kq['global_evidence']}")
        print(f"scoped_evidence={kq['scoped_evidence']}")
        print(f"out_jsonl={kq['out_jsonl_path']}")
        print(f"report_out={kq['report_path']}")

    elif args.command == "retrieve_vector":
        chunks_path = (
            args.chunks
            if args.chunks.is_absolute()
            else (project_root / args.chunks)
        )
        eval_q_path = (
            args.eval_questions
            if args.eval_questions.is_absolute()
            else (project_root / args.eval_questions)
        )
        out_jsonl = (
            args.out_jsonl
            if args.out_jsonl.is_absolute()
            else (project_root / args.out_jsonl)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        persist_dir = (
            args.persist_dir
            if args.persist_dir.is_absolute()
            else (project_root / args.persist_dir)
        )
        bm25_jsonl = default_processed_dir / "retrieval_bm25_dev_fixed.jsonl"
        kq = chay_danh_gia_vector_baseline(
            chunks_path=chunks_path,
            eval_questions_path=eval_q_path,
            out_jsonl_path=out_jsonl,
            report_path=report_out,
            persist_dir=persist_dir,
            model_name=args.model_name,
            top_k=args.top_k,
            bm25_jsonl_path=bm25_jsonl,
        )
        print("===== RETRIEVAL VECTOR BASELINE (BGE + CHROMADB) =====")
        print(
            f"n_chunks={kq['n_chunks']} n_questions={kq['n_questions']} "
            f"n_questions_with_gt={kq['n_questions_with_gt']} "
            f"index_time_sec={kq['index_time_sec']} total_time_sec={kq['total_time_sec']}"
        )
        print(f"global_asin={kq['global_asin']}")
        print(f"global_evidence={kq['global_evidence']}")
        print(f"scoped_evidence={kq['scoped_evidence']}")
        print(f"out_jsonl={kq['out_jsonl_path']}")
        print(f"report_out={kq['report_path']}")

    elif args.command == "generate_no_rag":
        eval_q_path = (
            args.eval_questions
            if args.eval_questions.is_absolute()
            else (project_root / args.eval_questions)
        )
        out_jsonl = (
            args.out_jsonl
            if args.out_jsonl.is_absolute()
            else (project_root / args.out_jsonl)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        kq = chay_generation_no_rag_baseline(
            eval_questions_path=eval_q_path,
            out_jsonl_path=out_jsonl,
            report_path=report_out,
            model_name=args.model_name,
            delay_sec=args.delay_sec,
        )
        print("===== GENERATION BASELINE 0 (NO-RAG) =====")
        print(
            f"n_questions={kq['n_questions']} n_direct_answer={kq['n_direct_answer']} "
            f"n_abstention={kq['n_abstention']} mean_latency_sec={kq['mean_latency_sec']} "
            f"total_tokens={kq['total_tokens']}"
        )
        print(f"by_label_3={kq['by_label_3']}")
        print(f"out_jsonl={kq['out_jsonl_path']}")
        print(f"report_out={kq['report_path']}")

    elif args.command == "generate_rag":
        ret_path = (
            args.retrieval_jsonl
            if args.retrieval_jsonl.is_absolute()
            else (project_root / args.retrieval_jsonl)
        )
        out_jsonl = (
            args.out_jsonl
            if args.out_jsonl.is_absolute()
            else (project_root / args.out_jsonl)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        no_rag_path = default_processed_dir / "generations_no_rag_dev.jsonl"
        kq = chay_generation_rag_baseline(
            retrieval_jsonl_path=ret_path,
            out_jsonl_path=out_jsonl,
            report_path=report_out,
            no_rag_jsonl_path=no_rag_path,
            model_name=args.model_name,
            top_k=args.top_k,
            delay_sec=args.delay_sec,
        )
        print("===== GENERATION BASELINE 1 (NAIVE RAG FIXED) =====")
        print(
            f"n_questions={kq['n_questions']} n_direct_answer={kq['n_direct_answer']} "
            f"n_abstention={kq['n_abstention']} mean_latency_sec={kq['mean_latency_sec']} "
            f"total_tokens={kq['total_tokens']}"
        )
        print(f"by_label_3={kq['by_label_3']}")
        print(f"out_jsonl={kq['out_jsonl_path']}")
        print(f"report_out={kq['report_path']}")

    elif args.command == "evaluate_judge":
        eval_labeled = (
            args.eval_labeled
            if args.eval_labeled.is_absolute()
            else (project_root / args.eval_labeled)
        )
        no_rag_gen = (
            args.no_rag_gen
            if args.no_rag_gen.is_absolute()
            else (project_root / args.no_rag_gen)
        )
        rag_gen = (
            args.rag_gen
            if args.rag_gen.is_absolute()
            else (project_root / args.rag_gen)
        )
        out_no_rag = (
            args.out_no_rag
            if args.out_no_rag.is_absolute()
            else (project_root / args.out_no_rag)
        )
        out_rag = (
            args.out_rag
            if args.out_rag.is_absolute()
            else (project_root / args.out_rag)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        kq = chay_danh_gia_judge_baseline(
            eval_labeled_path=eval_labeled,
            no_rag_gen_path=no_rag_gen,
            rag_gen_path=rag_gen,
            out_no_rag_judge_path=out_no_rag,
            out_rag_judge_path=out_rag,
            report_path=report_out,
            judge_model=args.judge_model,
            batch_size=args.batch_size,
        )
        print("===== EVALUATION JUDGE BASELINE (GEMINI + RAGAS + KAPPA) =====")
        print(f"summary_no_rag={kq['summary_no_rag']}")
        print(f"summary_rag={kq['summary_rag']}")
        print(f"kappa_judge_vs_human_strict={kq['kappa_judge_vs_human_strict']}")
        print(f"kappa_judge_vs_human_lenient={kq['kappa_judge_vs_human_lenient']}")
        print(f"kappa_rag_vs_human_strict={kq['kappa_rag_vs_human_strict']}")
        print(f"report_out={kq['report_path']}")

    elif args.command == "chunk_strategies":
        processed_dir = (
            args.processed_dir
            if args.processed_dir.is_absolute()
            else (project_root / args.processed_dir)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        embed_fn = None
        if "semantic" in args.strategies:
            embedder = BGEFastEmbedder(model_name=default_embedding_model, batch_size=32)
            embed_fn = embedder.embed_documents
        kq = chay_chunk_strategies(
            processed_dir=processed_dir,
            report_path=report_out,
            strategies=args.strategies,
            params={
                "max_words": default_chunk_size_words,
                "overlap_words": default_chunk_overlap_words,
                "child_max_words": 60,
            },
            embed_fn=embed_fn,
            split="dev",
        )
        print("===== CHUNK STRATEGIES (DEV) =====")
        print(f"report_out={kq['report_path']}")

    elif args.command == "retrieve_strategies":
        processed_dir = (
            args.processed_dir
            if args.processed_dir.is_absolute()
            else (project_root / args.processed_dir)
        )
        eval_q = (
            args.eval_questions
            if args.eval_questions.is_absolute()
            else (project_root / args.eval_questions)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        kq = chay_danh_gia_truy_xuat_6_chien_luoc(
            processed_dir=processed_dir,
            eval_questions_path=eval_q,
            report_path=report_out,
            strategies=args.strategies,
            top_k=default_top_k,
            reuse_existing=True,
        )
        print("===== RETRIEVE 6 CHUNKING STRATEGIES =====")
        print(f"report_out={kq['report_path']}")

    elif args.command == "rag_strategies":
        processed_dir = (
            args.processed_dir
            if args.processed_dir.is_absolute()
            else (project_root / args.processed_dir)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        kq = chay_rag_va_judge_cac_chien_luoc(
            processed_dir=processed_dir,
            report_path=report_out,
            strategies=args.strategies,
            generator_model=args.generator_model,
            judge_model=default_judge_model,
            top_k=default_top_k,
            gen_delay_sec=args.gen_delay,
            skip_judge=args.skip_judge,
            retrieval_prefix=args.retrieval,
            run_tag=args.run_tag,
        )
        print("===== RAG & JUDGE CHUNKING STRATEGIES =====")
        print(f"report_out={kq['report_path']}")

    elif args.command == "retrieve_hybrid":
        processed_dir = (
            args.processed_dir
            if args.processed_dir.is_absolute()
            else (project_root / args.processed_dir)
        )
        eval_q = (
            args.eval_questions
            if args.eval_questions.is_absolute()
            else (project_root / args.eval_questions)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        kq = chay_hybrid_6_chien_luoc(
            processed_dir=processed_dir,
            eval_questions_path=eval_q,
            report_path=report_out,
            strategies=args.strategies,
            k0=args.k0,
            depth=args.depth,
            top_k=default_top_k,
        )
        print("===== HYBRID RETRIEVAL RRF (BM25 + VECTOR) =====")
        print(f"report_out={kq['report_path']}")

    elif args.command == "build_policy_corpus":
        out_jsonl = (
            args.out_jsonl
            if args.out_jsonl.is_absolute()
            else (project_root / args.out_jsonl)
        )
        report_out = (
            args.report_out
            if args.report_out.is_absolute()
            else (project_root / args.report_out)
        )
        kq = xay_dung_kho_chinh_sach(
            out_jsonl_path=out_jsonl,
            report_path=report_out,
        )
        print("===== BUILD POLICY CORPUS (SHOPIFY LEGAL) =====")
        print(f"n_docs={kq['n_docs']} total_passages={kq['total_passages']} est_pages={kq['est_pages']}")
        print(f"out_jsonl={kq['out_jsonl_path']}")
        print(f"report_out={kq['report_path']}")

    elif args.command == "build_specialized_eval":
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else (project_root / args.processed_dir)
        eval_dir = args.eval_dir if args.eval_dir.is_absolute() else (project_root / args.eval_dir)
        report_dir = args.report_dir if args.report_dir.is_absolute() else (project_root / args.report_dir)

        kq_qa = xay_dung_bo_cau_hoi_chinh_sach(
            policy_docs_path=processed_dir / "policy_docs_shopify.jsonl",
            out_eval_path=eval_dir / "eval_policy_shopify.jsonl",
        )
        kq_ret = danh_gia_truy_xuat_chinh_sach(
            policy_docs_path=processed_dir / "policy_docs_shopify.jsonl",
            eval_policy_path=eval_dir / "eval_policy_shopify.jsonl",
            out_hybrid_jsonl_path=processed_dir / "retrieval_policy_hybrid_shopify.jsonl",
            report_path=report_dir / "policy_retrieval_evaluation.md",
            cache_dir=processed_dir / "emb_cache",
        )
        kq_cf = xay_dung_bo_counterfactual_30(
            eval_120_path=eval_dir / "eval_120_dev_labeled.jsonl",
            out_cf_path=eval_dir / "eval_counterfactual_30_dev.jsonl",
        )
        print("===== BUILD SPECIALIZED EVAL (POLICY + COUNTERFACTUAL) =====")
        print(f"policy_qa={kq_qa['n_questions']} counts={kq_qa['counts_by_category']}")
        print(f"policy_retrieval_summary={kq_ret['summary']}")
        print(f"counterfactual_n={kq_cf['n_counterfactual']}")

    elif args.command == "eval_policy_rag":
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else (project_root / args.processed_dir)
        report_out = args.report_out if args.report_out.is_absolute() else (project_root / args.report_out)
        kq = chay_rag_va_judge_chinh_sach(
            retrieval_policy_path=processed_dir / "retrieval_policy_hybrid_shopify.jsonl",
            out_gen_path=processed_dir / "generations_policy_shopify_ollama.jsonl",
            out_judge_path=processed_dir / "eval_judge_policy_shopify_ollama.jsonl",
            report_path=report_out,
            generator_model=args.generator_model,
            judge_model=default_judge_model,
            gen_delay_sec=args.gen_delay,
        )
        print("===== EVAL POLICY RAG (SHOPIFY 60 QA) =====")
        print(f"n_total={kq['n_total']} report_out={kq['report_path']}")

    elif args.command == "eval_counterfactual":
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else (project_root / args.processed_dir)
        eval_dir = args.eval_dir if args.eval_dir.is_absolute() else (project_root / args.eval_dir)
        report_out = args.report_out if args.report_out.is_absolute() else (project_root / args.report_out)
        kq = chay_danh_gia_counterfactual(
            cf_path=eval_dir / "eval_counterfactual_30_dev.jsonl",
            out_gen_path=processed_dir / "generations_counterfactual_30_dev_ollama.jsonl",
            out_judge_path=processed_dir / "eval_judge_counterfactual_30_dev_ollama.jsonl",
            report_path=report_out,
            generator_model=args.generator_model,
            judge_model=default_judge_model,
            gen_delay_sec=args.gen_delay,
        )
        print("===== EVAL COUNTERFACTUAL (30 QA) =====")
        print(f"adherence={kq['adherence_count']}/{kq['n_total']} ({kq['adherence_rate']}%) report_out={kq['report_path']}")

    elif args.command == "prepare_test_eval":
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else (project_root / args.processed_dir)
        eval_dir = args.eval_dir if args.eval_dir.is_absolute() else (project_root / args.eval_dir)
        report_out = args.report_out if args.report_out.is_absolute() else (project_root / args.report_out)
        eval_test_path = eval_dir / "eval_120_test_stratified.jsonl"
        kq_s = lay_mau_phan_tang_test_120(
            questions_test_clean_path=processed_dir / "questions_test_clean.jsonl",
            out_eval_path=eval_test_path,
            n_per_cell=30,
            seed=default_seed,
        )
        kq_r = chuan_bi_truy_xuat_test(
            processed_dir=processed_dir,
            eval_test_path=eval_test_path,
            report_path=report_out,
        )
        print("===== PREPARE TEST EVAL (120 HELD-OUT) =====")
        print(f"selected={kq_s['n_selected']} cells={kq_s['cell_counts']}")
        print(f"report_out={kq_r['report_path']}")

    elif args.command == "run_test_rag_judge":
        processed_dir = args.processed_dir if args.processed_dir.is_absolute() else (project_root / args.processed_dir)
        eval_dir = args.eval_dir if args.eval_dir.is_absolute() else (project_root / args.eval_dir)
        report_out = args.report_out if args.report_out.is_absolute() else (project_root / args.report_out)
        kq = chay_danh_gia_chot_tap_test(
            processed_dir=processed_dir,
            eval_test_path=eval_dir / "eval_120_test_stratified.jsonl",
            report_path=report_out,
            generator_model=args.generator_model,
            judge_model=default_judge_model,
            gen_delay_sec=args.gen_delay,
        )
        print("===== FINAL TEST EVALUATION (4 LOCKED CONFIGS) =====")
        print(f"report_out={kq['report_path']}")


if __name__ == "__main__":
    main()
