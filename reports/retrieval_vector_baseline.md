# Báo cáo Truy xuất Cơ sở bằng Dense Vector (`BAAI/bge-small-en-v1.5` + `ChromaDB`)

## 1. Cấu hình thực nghiệm
- Mô hình Embedding: **`BAAI/bge-small-en-v1.5`** (384 chiều, chuẩn hóa Cosine, thực thi qua ONNX Runtime / FastEmbed)
- Cơ sở dữ liệu Vector: **`ChromaDB`** (`hnsw:space = cosine`)
- Tập đoạn văn cơ sở: `chunks_dev_fixed.jsonl` (**2814 chunks** từ 150 sản phẩm `Electronics`, `fixed_size` 200 từ, overlap 40 từ)
- Tập câu hỏi đánh giá: `eval_120_dev_labeled.jsonl` (**120 câu hỏi**)
- Thời gian nhúng & lập chỉ mục 2814 chunks: **1500.71 giây** (Tổng thời gian chạy: **1519.02 giây**)

## 2. Kết quả Truy xuất Dense Vector (Hit@1, Hit@3, Hit@5, MRR@5)

| Chế độ truy xuất & Chỉ số đo | Số câu tính | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1. Toàn kho (Global — Định vị đúng sản phẩm `ASIN`)** | 120 | 0.3000 | 0.4250 | 0.4833 | 0.3671 |
| **2. Toàn kho (Global — Trúng đúng `chunk` chứa bằng chứng)** | 91 | 0.2857 | 0.4066 | 0.4945 | 0.3568 |
| **3. Trong trang sản phẩm (ASIN-scoped — Trúng `chunk` bằng chứng)** | 91 | **0.7912** | **0.9011** | **0.9451** | **0.8524** |

## 3. Phân rã kết quả ASIN-scoped theo nhóm nhãn chi tiết

| Nhóm nhãn chi tiết (`nhan_chot_chi_tiet`) | Số câu | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `answerable_full` | 27 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `partial_10_only` | 7 | 0.8571 | 1.0000 | 1.0000 | 0.9286 |
| `partial_expanded` | 38 | 0.8947 | 1.0000 | 1.0000 | 0.9474 |
| `recovered_in_full_doc` | 19 | 0.2632 | 0.5263 | 0.7368 | 0.4246 |

## 4. Đối sánh trực tiếp: BM25 (Lexical) vs. BAAI/bge-small-en-v1.5 (Dense Vector)

| Kịch bản đánh giá | Phương pháp | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Toàn kho (`Global` — Định vị đúng `ASIN`, 120 câu)** | BM25 (Từ khóa) | 0.4083 | 0.5083 | 0.5583 | 0.4590 |
| **Toàn kho (`Global` — Định vị đúng `ASIN`, 120 câu)** | **BGE-small-en-v1.5 (Vector)** | **0.3000** | **0.4250** | **0.4833** | **0.3671** |
| **Toàn kho (`Global` — Trúng `chunk` bằng chứng, 91 câu)** | BM25 (Từ khóa) | 0.4066 | 0.5165 | 0.5714 | 0.4603 |
| **Toàn kho (`Global` — Trúng `chunk` bằng chứng, 91 câu)** | **BGE-small-en-v1.5 (Vector)** | **0.2857** | **0.4066** | **0.4945** | **0.3568** |
| **Trong sản phẩm (`ASIN-scoped` — 91 câu có bằng chứng)** | BM25 (Từ khóa) | 0.8352 | 0.8681 | 0.9121 | 0.8566 |
| **Trong sản phẩm (`ASIN-scoped` — 91 câu có bằng chứng)** | **BGE-small-en-v1.5 (Vector)** | **0.7912** | **0.9011** | **0.9451** | **0.8524** |
| **Nhóm câu khó (`recovered_in_full_doc`, 19 câu)** | BM25 (Từ khóa) | 0.2105 | 0.3684 | 0.5789 | 0.3132 |
| **Nhóm câu khó (`recovered_in_full_doc`, 19 câu)** | **BGE-small-en-v1.5 (Vector)** | **0.2632** | **0.5263** | **0.7368** | **0.4246** |
