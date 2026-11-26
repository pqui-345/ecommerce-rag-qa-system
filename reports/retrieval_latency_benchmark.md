# Benchmark Hiệu năng Vận hành & Độ trễ Truy xuất (Dev 150 sản phẩm, 120 truy vấn)

- **Thiết bị đo:** CPU cục bộ (Windows x64, FastEmbed ONNX `BAAI/bge-small-en-v1.5`, NumPy float32 cosine).
- **Độ trễ mã hóa truy vấn đơn lẻ (Query Embedding):** Mean = **7.14 ms**, P50 = **5.81 ms**, P95 = **11.44 ms**.
- **Dung lượng kho ChromaDB hiện có (`data/vectordb`):** **32.36 MB**.

## 1. Tài nguyên lập chỉ mục theo chiến lược phân đoạn

| Chiến lược | Parent chunks | Indexed units | File chunks (MB) | Ma trận vector `.npy` (MB) | Thời gian dựng BM25 (s) | Thời gian nạp ma trận vector (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 2814 | 2814 | 3.65 | 4.12 | 0.236 | 0.078 |
| `recursive` | 2992 | 2992 | 3.22 | 4.38 | 0.356 | 0.206 |
| `parent_child` | 2895 | 9942 | 8.44 | 14.56 | 0.305 | 0.409 |

## 2. Độ trễ tìm kiếm trung bình trên mỗi truy vấn (ms/query, chưa tính encoding)

| Chiến lược | Phạm vi | Phương thức | Mean (ms) | P50 (ms) | P95 (ms) | Tổng E2E (Mean + Query Emb, ms) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `fixed` | `scoped` | BM25 | 0.05 | 0.04 | 0.11 | 0.05 |
| `fixed` | `scoped` | Dense Vector (search) | 0.05 | 0.03 | 0.06 | 7.19 |
| `fixed` | `scoped` | Hybrid RRF (search) | 0.10 | 0.08 | 0.17 | 7.24 |
| `fixed` | `global` | BM25 | 6.19 | 4.92 | 13.70 | 6.19 |
| `fixed` | `global` | Dense Vector (search) | 0.86 | 0.65 | 1.76 | 8.00 |
| `fixed` | `global` | Hybrid RRF (search) | 7.42 | 6.15 | 15.25 | 14.56 |
| `recursive` | `scoped` | BM25 | 0.10 | 0.07 | 0.28 | 0.10 |
| `recursive` | `scoped` | Dense Vector (search) | 0.09 | 0.07 | 0.26 | 7.23 |
| `recursive` | `scoped` | Hybrid RRF (search) | 0.21 | 0.15 | 0.55 | 7.35 |
| `recursive` | `global` | BM25 | 8.37 | 7.12 | 18.15 | 8.37 |
| `recursive` | `global` | Dense Vector (search) | 1.02 | 0.85 | 2.03 | 8.16 |
| `recursive` | `global` | Hybrid RRF (search) | 9.76 | 8.25 | 20.70 | 16.90 |
| `parent_child` | `scoped` | BM25 | 0.21 | 0.14 | 0.47 | 0.21 |
| `parent_child` | `scoped` | Dense Vector (search) | 0.16 | 0.12 | 0.42 | 7.30 |
| `parent_child` | `scoped` | Hybrid RRF (search) | 0.34 | 0.23 | 1.00 | 7.48 |
| `parent_child` | `global` | BM25 | 18.00 | 15.91 | 33.01 | 18.00 |
| `parent_child` | `global` | Dense Vector (search) | 2.35 | 2.04 | 3.95 | 9.49 |
| `parent_child` | `global` | Hybrid RRF (search) | 20.76 | 17.68 | 36.62 | 27.90 |
