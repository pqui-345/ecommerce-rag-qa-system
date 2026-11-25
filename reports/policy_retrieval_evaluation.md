# Báo cáo Đánh giá Truy xuất Chính sách TMĐT (Nguồn dữ liệu thứ 2 — Shopify Legal)

## 1. Cấu hình thực nghiệm (Báo cáo tách riêng khỏi AmazonQA)
- Kho tài liệu: `policy_docs_shopify.jsonl` (**27 tài liệu**, **527 passages** chuẩn có `gold_passage_id`)
- Tập câu hỏi chuyên biệt: `eval_policy_shopify.jsonl` (**60 câu hỏi**: 20 `factoid`, 15 `multi_condition`, 10 `ambiguous`, 15 `unanswerable`)
- Chỉ số đo trực tiếp trên **45 câu có `gold_passage_id`**: **Recall@1, Recall@3, Recall@5, MRR@5** đúng nghĩa (không dùng xấp xỉ)

## 2. Kết quả Tổng hợp (45 câu có `gold_passage_id`)

| Phương pháp truy xuất | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| :--- | ---: | ---: | ---: | ---: |
| Okapi BM25 (`k1=1.5, b=0.75`) | 0.4963 | 0.7093 | **0.7481** | **0.7259** |
| Dense Vector (`BAAI/bge-small-en-v1.5`) | 0.4926 | 0.6648 | **0.7333** | **0.6970** |
| **Hybrid RRF (`k0=60`, `depth=50`)** | 0.5370 | 0.7556 | **0.8148** | **0.7489** |

## 3. Phân rã Recall@5 và MRR@5 theo Loại câu hỏi chuyên biệt

| Loại câu hỏi | Số câu | BM25 (Recall@5 / MRR@5) | Dense Vector (Recall@5 / MRR@5) | **Hybrid RRF (Recall@5 / MRR@5)** |
| :--- | ---: | :---: | :---: | :---: |
| `factoid` | 20 | 0.9250 / 0.8417 | 0.9000 / 0.8100 | **0.9500 / 0.8750** |
| `multi_condition` | 15 | 0.7778 / 0.8500 | 0.7333 / 0.8556 | **0.8556 / 0.8444** |
| `ambiguous` | 10 | 0.3500 / 0.3083 | 0.4000 / 0.2333 | **0.4833 / 0.3533** |
