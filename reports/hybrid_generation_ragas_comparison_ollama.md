# Báo cáo Đối sánh Sinh câu trả lời & RAGAS giữa các Chiến lược Phân đoạn (RQ1 & RQ2)

## 1. Cấu hình hệ thống cố định
- **Mô hình Embedding & Truy xuất:** `BAAI/bge-small-en-v1.5` + `ChromaDB` (Cosine, `ASIN-scoped Top-5`)
- **Mô hình Sinh câu trả lời (Generator):** `qwen/qwen3.8-27b` (Groq API, `temperature = 0.0`, `max_tokens = 256`)
- **Mô hình Giám khảo độc lập (Judge):** `gemini-3.5-flash` (Google Gemini API, `temperature = 0.0`)
- **Tập đánh giá:** `eval_120_dev_labeled.jsonl` (**120 câu hỏi**: 27 `yes`, 64 `partial`, 29 `no`/bẫy)

## 2. Hành vi Trả lời theo Nhóm Nhãn Chuẩn (`yes` / `partial` / `no`)

| Chiến lược | Tổng trả lời trực tiếp (/120) | Tỷ lệ trả lời nhóm `yes` (/27) | Tỷ lệ trả lời nhóm `partial` (/64) | Tỷ lệ chặn câu bẫy `no` (/29) |
| :--- | ---: | ---: | ---: | ---: |
| **`fixed`** | 52/120 (43.33%) | 25/27 (**92.59%**) | 27/64 (**42.19%**) | 29/29 (**100.00%**) |
| **`recursive`** | 53/120 (44.17%) | 27/27 (**100.00%**) | 26/64 (**40.62%**) | 29/29 (**100.00%**) |
| **`parent_child`** | 53/120 (44.17%) | 25/27 (**92.59%**) | 28/64 (**43.75%**) | 29/29 (**100.00%**) |

## 3. Các Chỉ số RAGAS, Độ Trung thực, Ảo giác & Đồng thuận Giám khảo (Cohen's $\kappa$)

| Chiến lược | CP@5 (`yes`) | CP@5 (`yes+partial`) | Faithfulness (Direct) | Faithfulness (All) | Hallucination on Direct | Answer Rel (Direct) | Answer Rel (`yes+partial`) | $\kappa$ (Judge vs. RAG) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fixed`** | 0.8662 | **0.6157** | **0.9865** | 0.9942 | **1.92%** (1/52) | 0.9577 | **0.5473** | 0.8991 |
| **`recursive`** | 0.8831 | **0.6646** | **0.9811** | 0.9917 | **0.00%** (0/53) | 0.9604 | **0.5593** | 0.8994 |
| **`parent_child`** | 0.8823 | **0.6605** | **0.9906** | 0.9958 | **1.89%** (1/53) | 0.9481 | **0.5522** | 0.8664 |

## 4. Kiểm định McNemar so với Baseline `fixed` trên Nhóm `partial` (64 câu) & Hành vi Đúng Toàn tập (120 câu)

| Chiến lược so với `fixed` | $\Delta$ Câu trả lời `partial` | McNemar trên `partial` ($b / c$, $p$-value) | McNemar trên Hành vi đúng không ảo giác ($b / c$, $p$-value) |
| :--- | ---: | :--- | :--- |
| `recursive` vs. `fixed` | -1 câu (-1.57%) | `b=4, c=3` ($p=1.0000$) | `b=4, c=6` ($p=0.7539$) |
| `parent_child` vs. `fixed` | +1 câu (+1.56%) | `b=2, c=3` ($p=1.0000$) | `b=3, c=4` ($p=1.0000$) |
