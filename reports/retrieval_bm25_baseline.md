# Báo cáo Truy xuất Cơ sở bằng BM25 (Lexical Baseline Retrieval)

## 1. Cấu hình thực nghiệm
- Tập đoạn văn cơ sở: `chunks_dev_fixed.jsonl` (**2814 chunks** từ 150 sản phẩm `Electronics`, `fixed_size` 200 từ, overlap 40 từ)
- Tập câu hỏi đánh giá: `eval_120_dev_labeled.jsonl` (**120 câu hỏi** đã kiểm định nhãn 2 vòng)
- Thuật toán truy xuất: **Okapi BM25** (`k1=1.5`, `b=0.75`, tách từ alphanumeric chữ thường, loại bỏ stopword tiếng Anh)
- Số đoạn truy xuất (`top_k`): **5**
- Số câu hỏi có bằng chứng trong tài liệu (`yes` + `partial` khớp chunk đích): **91 / 91 câu** (29 câu còn lại thuộc nhóm `unanswerable_in_doc` không có đoạn đích trong tài liệu)

## 2. Kết quả Truy xuất Tổng hợp (Hit@1, Hit@3, Hit@5, MRR@5)

| Chế độ truy xuất & Chỉ số đo | Số câu tính | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1. Toàn kho (Global — Định vị đúng sản phẩm `ASIN`)** | 120 | 0.4083 | 0.5083 | 0.5583 | 0.4590 |
| **2. Toàn kho (Global — Trúng đúng `chunk` chứa bằng chứng)** | 91 | 0.4066 | 0.5165 | 0.5714 | 0.4603 |
| **3. Trong trang sản phẩm (ASIN-scoped — Trúng `chunk` bằng chứng)** | 91 | **0.8352** | **0.8681** | **0.9121** | **0.8566** |

## 3. Phân rã kết quả ASIN-scoped theo nhóm nhãn chi tiết

| Nhóm nhãn chi tiết (`nhan_chot_chi_tiet`) | Số câu | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `answerable_full` | 27 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `partial_10_only` | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `partial_expanded` | 38 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `recovered_in_full_doc` | 19 | 0.2105 | 0.3684 | 0.5789 | 0.3132 |
