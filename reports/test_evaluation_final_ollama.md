# Báo cáo Đánh giá Chốt 1 Lần Duy Nhất trên Tập Test Độc Lập (120 câu Electronics)

- **Generator:** `qwen3.8:27b` | **Judge:** `gemini-3.5-flash` | **Tập đánh giá:** `eval_120_test_stratified.jsonl` (150 `asin` cô lập hoàn toàn với `dev`)

| Cấu hình đã khóa | Trả lời trực tiếp (/120) | CP@5 (All) | CP@5 (`judge_ans=1`) | Faithfulness (Direct) | Faithfulness (All) | Hallucination on Direct | Answer Rel (Direct) | Answer Rel (All) | $\kappa$ (Judge vs RAG) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`no_rag`** | 13/120 (10.83%) | 0.0000 | 0.0000 | **0.5385** | 0.9500 | **30.77%** (4/13) | 0.6000 | 0.0650 | 0.0000 |
| **`fixed_vector`** | 56/120 (46.67%) | 0.4644 | 0.7858 | **1.0000** | 1.0000 | **0.00%** (0/56) | 0.9643 | 0.4500 | 0.9166 |
| **`recursive_hybrid`** | 52/120 (43.33%) | 0.4501 | 0.8163 | **1.0000** | 1.0000 | **0.00%** (0/52) | 0.9673 | 0.4192 | 0.8826 |
| **`parent_child_hybrid`** | 56/120 (46.67%) | 0.4402 | 0.7818 | **0.9911** | 0.9958 | **1.79%** (1/56) | 0.9750 | 0.4550 | 0.9666 |
