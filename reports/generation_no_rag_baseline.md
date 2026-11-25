# Báo cáo Sinh câu trả lời Đối chứng Không Ngữ cảnh (Baseline 0: No-RAG)

## 1. Cấu hình thực nghiệm
- Chế độ: **`no_rag` (LLM-only, không cung cấp đoạn đánh giá sản phẩm)**
- Mô hình Generator: **`qwen/qwen3.8-27b`** (Groq API, `temperature = 0.0`, `max_tokens = 256`)
- Tập câu hỏi đánh giá: `eval_120_dev_labeled.jsonl` (**120 câu hỏi**)
- Thời gian phản hồi trung bình: **1.907 giây / câu**
- Tổng số token tiêu thụ: **20012 tokens** (Độ dài câu trả lời trung vị: **34.0 từ**)

## 2. Thống kê hành vi trả lời của LLM khi Không có RAG (Phục vụ RQ2)

| Nhãn thực tế trong tài liệu (`nhan_chot_3_muc`) | Tổng số câu | LLM tự trả lời trực tiếp (`Direct Answer`) | LLM từ chối trả lời (`Abstention`) | Tỷ lệ tự trả lời không cần ngữ cảnh |
| :--- | ---: | ---: | ---: | ---: |
| **`yes`** | 27 | 1 | 26 | 3.70% |
| **`partial`** | 64 | 5 | 59 | 7.81% |
| **`no`** | 29 | 2 | 27 | 6.90% |
| **Tổng cộng toàn tập** | **120** | **8** | **112** | **6.67%** |
