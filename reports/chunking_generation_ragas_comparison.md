# Báo cáo Đối sánh Sinh câu trả lời & RAGAS giữa các Chiến lược Phân đoạn (RQ1 & RQ2)

## 1. Cấu hình hệ thống cố định
- **Mô hình Embedding & Truy xuất:** `BAAI/bge-small-en-v1.5` + `ChromaDB` (Cosine, `ASIN-scoped Top-5`)
- **Mô hình Sinh câu trả lời (Generator):** `qwen/qwen3.8-27b` (Groq API, `temperature = 0.0`, `max_tokens = 256`)
- **Mô hình Giám khảo độc lập (Judge):** `gemini-3.5-flash` (Google Gemini API, `temperature = 0.0`)
- **Tập đánh giá:** `eval_120_dev_labeled.jsonl` (**120 câu hỏi**: 27 `yes`, 64 `partial`, 29 `no`/bẫy)

## 2. Hành vi Trả lời theo Nhóm Nhãn Chuẩn (`yes` / `partial` / `no`)

| Chiến lược | Tổng trả lời trực tiếp (/120) | Tỷ lệ trả lời nhóm `yes` (/27) | Tỷ lệ trả lời nhóm `partial` (/64) | Tỷ lệ chặn câu bẫy `no` (/29) |
| :--- | ---: | ---: | ---: | ---: |
| **`fixed`** | 48/120 (40.00%) | 25/27 (**92.59%**) | 23/64 (**35.94%**) | 29/29 (**100.00%**) |
| **`recursive`** | 50/120 (41.67%) | 25/27 (**92.59%**) | 25/64 (**39.06%**) | 29/29 (**100.00%**) |
| **`sentence`** | 46/120 (38.33%) | 25/27 (**92.59%**) | 21/64 (**32.81%**) | 29/29 (**100.00%**) |
| **`structure`** | 47/120 (39.17%) | 25/27 (**92.59%**) | 22/64 (**34.38%**) | 29/29 (**100.00%**) |
| **`semantic`** | 45/120 (37.50%) | 24/27 (**88.89%**) | 21/64 (**32.81%**) | 29/29 (**100.00%**) |
| **`parent_child`** | 52/120 (43.33%) | 25/27 (**92.59%**) | 27/64 (**42.19%**) | 29/29 (**100.00%**) |

## 3. Các Chỉ số RAGAS, Độ Trung thực, Ảo giác & Đồng thuận Giám khảo (Cohen's $\kappa$)

| Chiến lược | CP@5 (`yes`) | CP@5 (`yes+partial`) | Faithfulness (Direct) | Faithfulness (All) | Hallucination on Direct | Answer Rel (Direct) | Answer Rel (`yes+partial`) | $\kappa$ (Judge vs. RAG) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fixed`** | 0.7984 | **0.5367** | **0.9854** | 0.9942 | **2.08%** (1/48) | 0.9917 | **0.5231** | 0.8814 |
| **`recursive`** | 0.8439 | **0.6160** | **0.9960** | 0.9983 | **0.00%** (0/50) | 0.9740 | **0.5352** | 0.8989 |
| **`sentence`** | 0.8252 | **0.5337** | **0.9957** | 0.9983 | **0.00%** (0/46) | 0.9609 | **0.4857** | 0.8960 |
| **`structure`** | 0.8360 | **0.5533** | **0.9574** | 0.9833 | **4.26%** (2/47) | 0.9447 | **0.4879** | 0.8959 |
| **`semantic`** | 0.8151 | **0.5328** | **0.9889** | 0.9958 | **2.22%** (1/45) | 0.9589 | **0.4742** | 0.8283 |
| **`parent_child`** | 0.8260 | **0.6194** | **1.0000** | 1.0000 | **0.00%** (0/52) | 0.9558 | **0.5462** | 0.8497 |

## 4. Kiểm định McNemar so với Baseline `fixed` trên Nhóm `partial` (64 câu) & Hành vi Đúng Toàn tập (120 câu)

| Chiến lược so với `fixed` | $\Delta$ Câu trả lời `partial` | McNemar trên `partial` ($b / c$, $p$-value) | McNemar trên Hành vi đúng không ảo giác ($b / c$, $p$-value) |
| :--- | ---: | :--- | :--- |
| `recursive` vs. `fixed` | +2 câu (+3.12%) | `b=4, c=6` ($p=0.7539$) | `b=3, c=6` ($p=0.5078$) |
| `sentence` vs. `fixed` | -2 câu (-3.13%) | `b=7, c=5` ($p=0.7744$) | `b=6, c=5` ($p=1.0000$) |
| `structure` vs. `fixed` | -1 câu (-1.56%) | `b=6, c=5` ($p=1.0000$) | `b=5, c=3` ($p=0.7266$) |
| `semantic` vs. `fixed` | -2 câu (-3.13%) | `b=5, c=3` ($p=0.7266$) | `b=6, c=3` ($p=0.5078$) |
| `parent_child` vs. `fixed` | +4 câu (+6.25%) | `b=2, c=6` ($p=0.2891$) | `b=3, c=8` ($p=0.2266$) |
