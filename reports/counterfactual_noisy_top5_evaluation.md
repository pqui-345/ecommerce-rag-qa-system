# Báo cáo Counterfactual Đa đoạn có Nhiễu (Noisy Top-5 Context — 30 câu)

- **Generator:** `qwen/qwen3.8-27b` | **Judge:** `openai/gpt-oss-120b` (Groq API)
- **Thiết lập:** 1 đoạn phản thực tế ghép ngẫu nhiên (`seed=42`) cùng 4 đoạn review thật cùng ASIN.
- **Số câu kiểm thử:** **30 câu**
- **Tuân thủ trọn vẹn ngữ cảnh phản thực tế (Strict CF Adherence):** **22/30 (73.33%)**
- **Ghi nhận mâu thuẫn đa nguồn / trả lời hai chiều (Conflict Hedging):** **3/30 (10.00%)**
- **Ngả theo nhận xét thật đối nghịch trong Top-5 / ký ức cũ:** **5/30 (16.67%)**
