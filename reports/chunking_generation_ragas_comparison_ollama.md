# Báo cáo Đối sánh Sinh câu trả lời & RAGAS giữa các Chiến lược Phân đoạn (RQ1 & RQ2)

## 1. Cấu hình hệ thống cố định
- **Mô hình Embedding & Truy xuất:** `BAAI/bge-small-en-v1.5` + `ChromaDB` (Cosine, `ASIN-scoped Top-5`)
- **Mô hình Sinh câu trả lời (Generator):** `qwen/qwen3.8-27b` (Groq API, `temperature = 0.0`, `max_tokens = 256`)
- **Mô hình Giám khảo độc lập (Judge):** `gemini-3.5-flash` (Google Gemini API, `temperature = 0.0`)
- **Tập đánh giá:** `eval_120_dev_labeled.jsonl` (**120 câu hỏi**: 27 `yes`, 64 `partial`, 29 `no`/bẫy)

## 2. Hành vi Trả lời theo Nhóm Nhãn Chuẩn (`yes` / `partial` / `no`)

| Chiến lược | Tổng trả lời trực tiếp (/120) | Tỷ lệ trả lời nhóm `yes` (/27) | Tỷ lệ trả lời nhóm `partial` (/64) | Tỷ lệ chặn câu bẫy `no` (/29) |
| :--- | ---: | ---: | ---: | ---: |
| **`fixed`** | 51/120 (42.50%) | 26/27 (**96.30%**) | 25/64 (**39.06%**) | 29/29 (**100.00%**) |
| **`recursive`** | 49/120 (40.83%) | 25/27 (**92.59%**) | 24/64 (**37.50%**) | 29/29 (**100.00%**) |
| **`sentence`** | 47/120 (39.17%) | 27/27 (**100.00%**) | 20/64 (**31.25%**) | 29/29 (**100.00%**) |
| **`structure`** | 48/120 (40.00%) | 25/27 (**92.59%**) | 23/64 (**35.94%**) | 29/29 (**100.00%**) |
| **`semantic`** | 49/120 (40.83%) | 25/27 (**92.59%**) | 24/64 (**37.50%**) | 29/29 (**100.00%**) |
| **`parent_child`** | 52/120 (43.33%) | 25/27 (**92.59%**) | 27/64 (**42.19%**) | 29/29 (**100.00%**) |

## 3. Các Chỉ số RAGAS, Độ Trung thực, Ảo giác & Đồng thuận Giám khảo (Cohen's $\kappa$)

| Chiến lược | CP@5 (`yes`) | CP@5 (`yes+partial`) | Faithfulness (Direct) | Faithfulness (All) | Hallucination on Direct | Answer Rel (Direct) | Answer Rel (`yes+partial`) | $\kappa$ (Judge vs. RAG) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fixed`** | 0.7827 | **0.5660** | **0.9961** | 0.9983 | **1.96%** (1/51) | 0.9569 | **0.5363** | 0.8657 |
| **`recursive`** | 0.8507 | **0.6044** | **0.9959** | 0.9983 | **0.00%** (0/49) | 0.9469 | **0.5099** | 0.8318 |
| **`sentence`** | 0.8527 | **0.5409** | **0.9957** | 0.9983 | **0.00%** (0/47) | 0.9745 | **0.5033** | 0.9653 |
| **`structure`** | 0.8473 | **0.5634** | **0.9792** | 0.9917 | **0.00%** (0/48) | 0.9500 | **0.5011** | 0.8475 |
| **`semantic`** | 0.7805 | **0.5869** | **0.9959** | 0.9983 | **0.00%** (0/49) | 0.9561 | **0.5148** | 0.8812 |
| **`parent_child`** | 0.8613 | **0.6329** | **1.0000** | 1.0000 | **0.00%** (0/52) | 0.9635 | **0.5505** | 0.8497 |

## 4. Kiểm định McNemar so với Baseline `fixed` trên Nhóm `partial` (64 câu) & Hành vi Đúng Toàn tập (120 câu)

| Chiến lược so với `fixed` | $\Delta$ Câu trả lời `partial` | McNemar trên `partial` ($b / c$, $p$-value) | McNemar trên Hành vi đúng không ảo giác ($b / c$, $p$-value) |
| :--- | ---: | :--- | :--- |
| `recursive` vs. `fixed` | -1 câu (-1.56%) | `b=5, c=4` ($p=1.0000$) | `b=6, c=5` ($p=1.0000$) |
| `sentence` vs. `fixed` | -5 câu (-7.81%) | `b=7, c=2` ($p=0.1797$) | `b=7, c=4` ($p=0.5488$) |
| `structure` vs. `fixed` | -2 câu (-3.12%) | `b=6, c=4` ($p=0.7539$) | `b=7, c=5` ($p=0.7744$) |
| `semantic` vs. `fixed` | -1 câu (-1.56%) | `b=4, c=3` ($p=1.0000$) | `b=5, c=4` ($p=1.0000$) |
| `parent_child` vs. `fixed` | +2 câu (+3.13%) | `b=2, c=4` ($p=0.6875$) | `b=3, c=5` ($p=0.7266$) |
