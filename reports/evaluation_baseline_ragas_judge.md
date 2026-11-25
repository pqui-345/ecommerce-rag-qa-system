# Báo cáo Đánh giá Baseline bằng Khung RAGAS & Giám khảo LLM (`Gemini Flash`)

## 1. Cấu hình hệ thống Baseline & Khung đánh giá
- **Mô hình Embedding:** `BAAI/bge-small-en-v1.5` (384 chiều, Cosine similarity trên `ChromaDB`)
- **Chiến lược phân đoạn cơ sở (Baseline Chunking):** `Fixed-size` (200 từ/đoạn, overlap 40 từ -> 2.814 chunks trên tập Dev)
- **Mô hình Sinh câu trả lời (Generator):** `qwen/qwen3.8-27b` (Groq API, `temperature = 0.0`)
- **Mô hình Giám khảo độc lập (LLM-as-a-Judge):** `gemini-3.5-flash` (Google Gemini API, `temperature = 0.0`)
- **Tập đánh giá đối chuẩn:** `eval_120_dev_labeled.jsonl` (**120 câu hỏi** đã kiểm định tay 2 vòng: 27 câu `yes`, 64 câu `partial`, 29 câu `no`/bẫy)

## 2. Kết quả Định lượng Đối sánh: Baseline 0 (`No-RAG`) vs. Baseline 1 (`Naive RAG — Fixed-size`)

| Chỉ số đánh giá (RAGAS & Hallucination Metrics) | Baseline 0 (`No-RAG`) | Baseline 1 (`Naive RAG — Fixed-size`) | Mức cải thiện của RAG |
| :--- | ---: | ---: | ---: |
| **1. Số câu trả lời trực tiếp (`Direct Answers` / 120 câu)** | 8 (6.67%) | **48 (40.00%)** | **+40 câu** |
| **2. Faithfulness trung bình trên các câu trả lời trực tiếp** *(Độ trung thực)* | 0.2500 | **0.9854** | **+0.7354** |
| **3. Faithfulness toàn tập (120 câu, tính cả từ chối đúng)** | 0.9500 | **0.9942** | **+0.0442** |
| **4. Tỷ lệ Ảo giác khi đưa ra câu trả lời (`Hallucination Rate on Direct`)** | 75.00% (6/8 câu) | **2.08% (1/48 câu)** | **-72.92%** |
| **5. Answer Relevance trên các câu trả lời trực tiếp** | 0.7500 | **0.9917** | **+0.2417** |
| **6. Answer Relevance trên nhóm câu có đáp án đầy đủ (`yes`, 27 câu)** | 0.0000 | **0.9259** | **+0.9259** |
| **7. Answer Relevance trên toàn bộ 91 câu có thông tin (`yes` + `partial`)** | 0.0440 | **0.5231** | **+0.4791** |
| **8. Context Precision@5 trên nhóm `yes` (27 câu)** | 0.0000 | **0.7984** | **+0.7984** |
| **9. Context Precision@5 trên nhóm `yes` + `partial` (91 câu)** | 0.0000 | **0.5367** | **+0.5367** |
| **10. Tỷ lệ chặn đúng câu hỏi bẫy (`Trap Rejection Rate` trên 29 câu `no`)** | 93.10% | **100.00%** | **+6.90%** |

## 3. Kiểm định Độ tin cậy của Giám khảo LLM so với Nhãn người (Cohen's $\kappa$ — Trả lời RQ3)

| Cặp đối chiếu (N = 120 câu hỏi) | Accuracy | Precision | Recall | F1-Score | $P_o$ | $P_e$ | **Cohen's $\kappa$** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. Giám khảo Gemini (`judge_answerable`) vs. Nhãn người Nghiêm ngặt (`yes`)** | 0.7500 | 0.4727 | 0.9630 | 0.6341 | 0.7500 | 0.5229 | **0.4760** |
| **2. Giám khảo Gemini (`judge_answerable`) vs. Nhãn người Mở rộng (`yes+partial`)** | 0.7000 | 1.0000 | 0.6044 | 0.7534 | 0.7000 | 0.4785 | **0.4248** |
| **3. Hành vi trả lời của Naive RAG vs. Nhãn người Nghiêm ngặt (`yes`)** | 0.7917 | 0.5208 | 0.9259 | 0.6667 | 0.7917 | 0.5550 | **0.5318** |
| **4. Giám khảo Gemini (`judge_answerable`) vs. Hành vi trả lời của Naive RAG** | 0.9417 | 0.8727 | 1.0000 | 0.9320 | 0.9417 | 0.5083 | **0.8814** |
