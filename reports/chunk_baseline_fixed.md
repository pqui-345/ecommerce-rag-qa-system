# Báo cáo Phân đoạn Cơ sở (Baseline Fixed-size Chunking)

## 1. Cấu hình tham số
- Chiến lược: `fixed_size` (cửa sổ trượt theo ranh giới từ, không cắt đôi từ, bảo toàn `\n\n` nội bộ)
- Kích thước đoạn (`chunk_size_words`): **200 từ** (~250-280 tokens, tương thích hoàn toàn với giới hạn 512 tokens của `BAAI/bge-small-en-v1.5`)
- Độ gối đầu (`chunk_overlap_words`): **40 từ** (20.0%)
- Bước trượt (`step`): **160 từ**

## 2. Thống kê kết quả phân đoạn

| Chỉ số | Tập Dev (`chunks_dev_fixed.jsonl`) | Tập Test (`chunks_test_fixed.jsonl`) |
| :--- | ---: | ---: |
| Số tài liệu đầu vào (`docs_*_clean.jsonl`) | 150 | 150 |
| Số dòng lỗi JSON / tài liệu rỗng | 0 / 0 | 0 / 0 |
| **Tổng số đoạn văn (chunks) sinh ra** | **2814** | **2865** |
| Kiểm tra `chunk_id` duy nhất 100% | ĐẠT | ĐẠT |
| Số chunks / tài liệu (Min / Trung vị / TB / P90 / Max) | 6 / 16.0 / 18.8 / 36.0 / 55 | 6 / 16.5 / 19.1 / 35.2 / 51 |
| Số từ / chunk (Min / Trung vị / TB / P90 / Max) | 41 / 200.0 / 195.7 / 200.0 / 200 | 42 / 200.0 / 195.8 / 200.0 / 200 |
| Số ký tự / chunk (Min / Trung vị / TB / P90 / Max) | 211 / 1075.0 / 1059.0 / 1147.0 / 1466 | 221 / 1080.0 / 1063.9 / 1147.0 / 1487 |
