# E-Commerce RAG QA System

Hệ thống hỏi đáp thông minh cho lĩnh vực thương mại điện tử kết hợp **Large Language Model (LLM)** và **Retrieval-Augmented Generation (RAG)**, tập trung tối ưu hóa chiến lược phân đoạn văn bản (*chunking*), truy xuất lai (*Hybrid RRF*) và khung đánh giá độ trung thực nhằm định lượng khả năng triệt tiêu ảo giác.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/pytest-88%20passed-brightgreen?logo=pytest&logoColor=white)
![Embedding](https://img.shields.io/badge/Embedding-BAAI%2Fbge--small--en--v1.5-orange)
![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)

---

## Mục tiêu & Câu hỏi nghiên cứu

| Câu hỏi | Trọng tâm nghiên cứu |
| :--- | :--- |
| **RQ1 — Phân đoạn & Truy xuất** | So sánh 6 chiến lược chunking (`fixed`, `recursive`, `sentence`, `structure`, `semantic`, `parent_child`) trên nhánh Dense Vector và Hybrid RRF (`k0 = 60`). |
| **RQ2 — Triệt tiêu ảo giác** | Định lượng mức giảm tỷ lệ ảo giác và khả năng từ chối an toàn các câu hỏi không có bằng chứng so với đối chứng **No-RAG**. |
| **RQ3 — Khung đánh giá** | Đánh giá độ trung thực qua **RAGAS** (Faithfulness, Answer Relevance, Context Precision), đối chiếu nhãn tay hai vòng (Cohen's $\kappa$) và thí nghiệm phản thực tế (*Counterfactual*). |

---

## Kiến trúc hệ thống

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Dữ liệu & Tiền xử lý                                                 │
│    • AmazonQA v2 (184.909 dòng, khóa uid, cô lập 150 dev / 150 test)    │
│    • Kho chính sách Shopify Legal (27 tài liệu, 527 gold passages)      │
│    • Làm sạch lũy đẳng 5 bước (clean_text, bảo toàn ký hiệu kỹ thuật)   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Phân đoạn (6 chiến lược) & Lập chỉ mục kép                           │
│    • Dense Vector: BAAI/bge-small-en-v1.5 + ChromaDB                    │
│    • Lexical: Okapi BM25 (k1 = 1.5, b = 0.75)                           │
│    • Hợp nhất: Reciprocal Rank Fusion (k0 = 60, depth = 50)             │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Sinh câu trả lời & Kiểm định (No-RAG vs. RAG)                        │
│    • Generator: qwen3.8:27b (temperature = 0)                           │
│    • Judge: gemini-3.5-flash (RAGAS + Counterfactual + Cohen's kappa)   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kết quả thực nghiệm nổi bật

### 1. Đánh giá chốt trên Tập Test Độc lập (120 câu Electronics, cô lập ASIN)

| Cấu hình khóa từ `dev` | Global Hit@5 | Scoped MRR@5 | Trả lời trực tiếp | Faithfulness (Direct) | Tỷ lệ ảo giác |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `no_rag` (Đối chứng) | — | — | 13/120 (10.83%) | 0.5385 | **30.77%** (4/13) |
| `fixed_vector` (Baseline) | 0.5583 | 0.8950 | 56/120 (46.67%) | **1.0000** | **0.00%** (0/56) |
| **`recursive_hybrid` (Tối ưu)** | **0.6750** | **0.9500** | 52/120 (43.33%) | **1.0000** | **0.00%** (0/52) |
| `parent_child_hybrid` | 0.5917 | 0.9271 | 56/120 (46.67%) | 0.9911 | 1.79% (1/56) |

### 2. Kho chính sách Shopify Legal & Thí nghiệm Phản thực tế

- **Truy xuất chính sách (45 câu có `gold_passage_id`):** Hybrid RRF đạt **Recall@5 = 0.8148**, **MRR@5 = 0.7489** (vượt Okapi BM25 `0.7481` và Dense Vector `0.7333`).
- **Thí nghiệm Phản thực tế (30 câu):** Tỷ lệ tuân thủ ngữ cảnh mới đạt **30/30 (100.00%)**, tỷ lệ quay về ký ức cũ bằng **0.00%**.
- **Độ đồng thuận giám khảo (Test):** Cohen's $\kappa$ giữa `gemini-3.5-flash` và hành vi RAG đạt **0.8826 – 0.9666**.

---

## Cấu trúc mã nguồn

```text
ecommerce-rag-qa-system/
├── config.yaml                 # Cấu hình đường dẫn, mô hình và siêu tham số
├── requirements.txt            # Thư viện phụ thuộc (pyyaml, pytest, chromadb, fastembed)
├── colab_run_rag.ipynb         # Notebook chạy thực nghiệm đồng bộ trên Google Colab
├── src/
│   ├── run.py                  # CLI entry point cho toàn bộ pipeline
│   ├── merge.py                # Gộp streaming AmazonQA theo khóa uid
│   ├── working_set.py          # Cô lập ASIN và dựng tập làm việc dev/test
│   ├── clean.py                # Chuẩn hóa văn bản 5 bước lũy đẳng
│   ├── label_verification.py   # Tổng hợp kiểm định nhãn tay 2 vòng
│   ├── chunking.py             # 6 chiến lược phân đoạn (fixed -> parent_child)
│   ├── retrieval_bm25.py       # Truy xuất từ khóa Okapi BM25
│   ├── retrieval_vector.py     # Truy xuất ngữ nghĩa BGE-small + ChromaDB
│   ├── hybrid_retrieval.py     # Hợp nhất thứ hạng Hybrid RRF (k0 = 60)
│   ├── chunking_eval.py        # Đánh giá truy xuất 6 chiến lược
│   ├── chunking_rag_eval.py    # Đánh giá RAGAS & Judge cho các chiến lược
│   ├── generation_baseline.py  # Nhánh sinh đối chứng No-RAG và RAG baseline
│   ├── evaluation_judge.py     # Đánh giá Faithfulness & Answer Relevance
│   ├── policy_corpus.py        # Xây dựng kho văn bản chính sách Shopify Legal
│   ├── specialized_eval.py     # Đánh giá câu hỏi chuyên biệt & Counterfactual
│   ├── test_set_eval.py        # Đánh giá chốt 1 lần duy nhất trên tập test
│   └── demo_cli.py             # CLI demo trực tiếp kịch bản bảo vệ
├── tests/                      # Bộ kiểm thử tự động pytest (88/88 passed)
└── reports/                    # Bảng tổng hợp số liệu thực nghiệm (.md)
```

---

## Cài đặt & Sử dụng

### 1. Cài đặt môi trường

```bash
git clone https://github.com/pqui-345/ecommerce-rag-qa-system.git
cd ecommerce-rag-qa-system
pip install -r requirements.txt
```

### 2. Thực thi pipeline qua CLI

Mọi lệnh được chạy từ thư mục gốc thông qua `python -m src.run <lệnh>`:

```bash
# 1. Gộp dữ liệu thô & dựng bộ làm việc cô lập ASIN
python -m src.run gop
python -m src.run bo_lam_viec
python -m src.run lam_sach

# 2. Phân đoạn văn bản (6 chiến lược) & đánh giá truy xuất
python -m src.run chunk_strategies
python -m src.run retrieve_strategies
python -m src.run retrieve_hybrid

# 3. Đánh giá sinh & kiểm định RAGAS
python -m src.run rag_strategies

# 4. Kho chính sách Shopify Legal & kiểm thử phản thực tế
python -m src.run build_policy_corpus
python -m src.run build_specialized_eval
python -m src.run eval_policy_rag
python -m src.run eval_counterfactual

# 5. Đánh giá chốt trên tập test độc lập (120 câu)
python -m src.run prepare_test_eval
python -m src.run run_test_rag_judge
```

### 3. Demo trực tiếp trên Terminal (CLI)

Chạy toàn bộ kịch bản demo 5 bước (tiền xử lý $\rightarrow$ truy xuất BM25 trực tiếp trên CPU $\rightarrow$ đối chiếu `No-RAG` vs. `RAG` $\rightarrow$ thí nghiệm phản thực tế $\rightarrow$ tổng hợp kết quả tập `test`):

```bash
# Chạy toàn bộ 5 bước demo
python -m src.demo_cli

# Hoặc chạy riêng từng bước (1..5)
python -m src.demo_cli --step 2

# Truy vấn thử trực tiếp trên dữ liệu Amazon Electronics hoặc Shopify Legal
python -m src.demo_cli --query "110 or 220 volt adapter" --source amazon --asin B002RYYZZS
python -m src.demo_cli --query "What financial remedies does Shopify provide for chargebacks?" --source shopify --top-k 2

# Chế độ hỏi đáp trực tiếp (Live RAG: truy xuất CPU + gọi LLM sinh No-RAG vs. RAG, có dự phòng offline)
python -m src.demo_cli --live --source amazon --asin B002RYYZZS --query "Is this adapter universal 110V and 220V?"
python -m src.demo_cli --live --source shopify
```

### 4. Chạy kiểm thử tự động

```bash
pytest -q
```

---

## Lưu ý dữ liệu

Thư mục `data/` và `logs/` được loại trừ khỏi Git theo `.gitignore`. Để tái lập toàn bộ pipeline từ đầu, đặt hai tệp dữ liệu gốc của **AmazonQA** (`val-qar.jsonl`, `test-qar.jsonl`) vào `data/raw/` trước khi chạy lệnh `gop`.
