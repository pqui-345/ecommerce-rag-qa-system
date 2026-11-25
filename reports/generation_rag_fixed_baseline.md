# Báo cáo Sinh câu trả lời Cơ sở Có Ngữ cảnh (Baseline 1: Naive RAG — Fixed-size)

## 1. Cấu hình thực nghiệm
- Chế độ: **`naive_rag_fixed` (Fixed-size 200 từ + `BAAI/bge-small-en-v1.5` + `ChromaDB` Top-5)**
- Mô hình Generator: **`qwen/qwen3.8-27b`** (Groq API, `temperature = 0.0`, `max_tokens = 256`)
- Tập dữ liệu truy xuất đầu vào: `retrieval_vector_dev_fixed.jsonl` (**120 câu hỏi**)
- Thời gian phản hồi trung bình: **3.076 giây / câu**
- Tổng số token tiêu thụ: **182665 tokens** (Độ dài câu trả lời trung vị: **31.5 từ**)

## 2. Thống kê hành vi trả lời của LLM khi Có RAG (Baseline 1)

| Nhãn thực tế trong tài liệu (`nhan_chot_3_muc`) | Tổng số câu | LLM trả lời trực tiếp (`Direct Answer`) | LLM từ chối trả lời (`Abstention`) | Tỷ lệ cung cấp câu trả lời (`Answer Rate`) | Tỷ lệ từ chối đúng câu bẫy (`Rejection Rate`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`yes`** | 27 | 25 | 2 | 92.59% | 7.41% |
| **`partial`** | 64 | 23 | 41 | 35.94% | 64.06% |
| **`no`** | 29 | 0 | 29 | 0.00% | 100.00% |
| **Tổng cộng toàn tập** | **120** | **48** | **72** | **40.00%** | **60.00%** |

## 3. Đối sánh trực tiếp: Baseline 0 (No-RAG) vs. Baseline 1 (Naive RAG — Fixed-size)

| Nhóm câu hỏi | Chỉ số | Baseline 0 (`No-RAG`) | Baseline 1 (`Naive RAG`) | Mức thay đổi |
| :--- | :--- | ---: | ---: | ---: |
| **Câu có đáp án đầy đủ (`yes`, 27 câu)** | Tỷ lệ trả lời trực tiếp | 1/27 (3.70%) | **25/27 (92.59%)** | **+88.89%** |
| **Câu có thông tin một phần (`partial`, 64 câu)** | Tỷ lệ trả lời trực tiếp | 5/64 (7.81%) | **23/64 (35.94%)** | **+28.12%** |
| **Câu không có thông tin / Bẫy (`no`, 29 câu)** | Tỷ lệ trả lời trực tiếp | 2/29 (6.90%) | **0/29 (0.00%)** | **-6.90%** |
