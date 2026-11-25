# Báo cáo Đánh giá RAG trên Kho Chính sách TMĐT (Shopify Legal — 60 câu chuyên biệt)

- **Generator:** `qwen3.8:27b` | **Judge:** `gemini-3.5-flash` | **Retrieval:** `Hybrid RRF Top-5`

| Nhóm câu hỏi | Số câu | Trả lời trực tiếp | Từ chối (Abstention) | CP@5 | Faithfulness (Direct) | Hallucination (Direct) | Answer Relevance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `factoid` | 20 | 19 (95.0%) | 1 (5.0%) | 0.8661 | 1.0000 | 0.00% | 0.9500 |
| `multi_condition` | 15 | 14 (93.3%) | 1 (6.7%) | 0.9082 | 1.0000 | 0.00% | 0.9333 |
| `ambiguous` | 10 | 10 (100.0%) | 0 (0.0%) | 0.8696 | 1.0000 | 0.00% | 0.9100 |
| `unanswerable` | 15 | 2 (13.3%) | 13 (86.7%) | 0.6386 | 1.0000 | 0.00% | 0.1333 |

- **Tỷ lệ nhận diện & làm rõ sự mơ hồ (`ambiguous`):** **1/10 (10.00%)**
