# Báo cáo Đối sánh Hiệu năng Truy xuất giữa 6 Chiến lược Phân đoạn (RQ1)

## 1. Cấu hình thực nghiệm đối chứng
- Tập tài liệu: `docs_dev_clean.jsonl` (**150 sản phẩm Electronics**, ngân sách trần **200 từ/đoạn**)
- Tập câu hỏi đánh giá: `eval_120_dev_labeled.jsonl` (**120 câu hỏi** kiểm tay 2 vòng: 27 `yes`, 64 `partial`, 29 `no`)
- Hai nhánh truy xuất độc lập: **Okapi BM25** (`k1=1.5, b=0.75`) và **Dense Vector `BAAI/bge-small-en-v1.5`** (ChromaDB Cosine, `top_k=5`)

## 2. Kết quả Truy xuất trong Trang Sản phẩm (`ASIN-scoped Evidence`, 91 câu có bằng chứng)

### 2.1. Nhánh Dense Vector (`BAAI/bge-small-en-v1.5` + `ChromaDB`)

| Chiến lược | Số chunks | Số câu có GT | Từ ngữ cảnh Top-5 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Nhóm câu khó `recovered` (Hit@5 / MRR@5) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fixed`** | 2814 | 91 | 977.6 | 0.7912 | 0.9011 | **0.9451** | **0.8524** | 0.7368 / 0.4246 |
| **`recursive`** | 2992 | 91 | 782.1 | 0.6593 | 0.8571 | **0.9011** | **0.7632** | 0.5263 / 0.2781 |
| **`sentence`** | 2840 | 91 | 905.3 | 0.7253 | 0.8681 | **0.9121** | **0.8024** | 0.5789 / 0.3298 |
| **`structure`** | 2895 | 91 | 783.0 | 0.6703 | 0.8681 | **0.8901** | **0.7632** | 0.5263 / 0.2956 |
| **`semantic`** | 2886 | 91 | 791.5 | 0.7582 | 0.8791 | **0.9341** | **0.8258** | 0.6842 / 0.4974 |
| **`parent_child`** | 2895 | 91 | 810.4 | 0.7143 | 0.9011 | **0.9011** | **0.7967** | 0.5263 / 0.3421 |

### 2.2. Nhánh Từ khóa Okapi BM25 (`k1=1.5, b=0.75`)

| Chiến lược | Số chunks | Số câu có GT | Từ ngữ cảnh Top-5 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Nhóm câu khó `recovered` (Hit@5 / MRR@5) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fixed`** | 2814 | 91 | 991.7 | 0.8352 | 0.8681 | **0.9121** | **0.8566** | 0.5789 / 0.3132 |
| **`recursive`** | 2992 | 91 | 803.1 | 0.7912 | 0.8571 | **0.8901** | **0.8240** | 0.4737 / 0.1570 |
| **`sentence`** | 2840 | 91 | 925.4 | 0.8132 | 0.8571 | **0.9121** | **0.8423** | 0.5789 / 0.2447 |
| **`structure`** | 2895 | 91 | 804.6 | 0.7802 | 0.8462 | **0.8791** | **0.8148** | 0.4211 / 0.1395 |
| **`semantic`** | 2886 | 91 | 814.7 | 0.8022 | 0.8352 | **0.8681** | **0.8227** | 0.3684 / 0.1772 |
| **`parent_child`** | 2895 | 91 | 809.2 | 0.7692 | 0.8022 | **0.8681** | **0.7987** | 0.3684 / 0.0886 |

## 3. Kết quả Truy xuất Toàn kho (`Global Retrieval` — 120 câu định vị ASIN & 91 câu trúng chunk)

| Chiến lược | Phương pháp | Global ASIN Hit@1 | Global ASIN Hit@5 | Global ASIN MRR@5 | Global Chunk Hit@1 | Global Chunk Hit@5 | Global Chunk MRR@5 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | Vector (`BGE-small`) | 0.3000 | 0.4833 | 0.3671 | 0.2857 | 0.4945 | 0.3568 |
| `recursive` | Vector (`BGE-small`) | 0.3333 | 0.4667 | 0.3772 | 0.2527 | 0.4615 | 0.3214 |
| `sentence` | Vector (`BGE-small`) | 0.3333 | 0.4833 | 0.3874 | 0.3077 | 0.4835 | 0.3722 |
| `structure` | Vector (`BGE-small`) | 0.3250 | 0.4500 | 0.3736 | 0.2308 | 0.4286 | 0.3090 |
| `semantic` | Vector (`BGE-small`) | 0.3083 | 0.4667 | 0.3699 | 0.2418 | 0.4945 | 0.3449 |
| `parent_child` | Vector (`BGE-small`) | 0.3083 | 0.5000 | 0.3792 | 0.2747 | 0.4945 | 0.3592 |
| `fixed` | BM25 | 0.4083 | 0.5583 | 0.4590 | 0.4066 | 0.5714 | 0.4603 |
| `recursive` | BM25 | 0.3667 | 0.5583 | 0.4358 | 0.3516 | 0.5495 | 0.4209 |
| `sentence` | BM25 | 0.3917 | 0.5833 | 0.4549 | 0.3846 | 0.5824 | 0.4522 |
| `structure` | BM25 | 0.3500 | 0.5500 | 0.4231 | 0.3297 | 0.5385 | 0.4040 |
| `semantic` | BM25 | 0.3750 | 0.5500 | 0.4431 | 0.3846 | 0.5714 | 0.4579 |
| `parent_child` | BM25 | 0.3000 | 0.5250 | 0.3789 | 0.2747 | 0.5275 | 0.3636 |

## 4. Kiểm định Ý nghĩa Thống kê so với Baseline `fixed` (Nhánh Vector ASIN-scoped)

| Chiến lược so với `fixed` | Số câu chung | $\Delta$ Hit@5 | McNemar Hit@5 ($b / c$, $p$-value) | $\Delta$ MRR@5 | Bootstrap 95% CI cho $\Delta$ MRR@5 |
| :--- | ---: | ---: | :--- | ---: | :--- |
| `recursive` vs. `fixed` | 91 | -0.0440 | `b=6, c=2` ($p=0.2891$) | -0.0892 | `[-0.1484, -0.0278]` |
| `sentence` vs. `fixed` | 91 | -0.0330 | `b=5, c=2` ($p=0.4531$) | -0.0500 | `[-0.1027, +0.0002]` |
| `structure` vs. `fixed` | 91 | -0.0549 | `b=7, c=2` ($p=0.1797$) | -0.0892 | `[-0.1557, -0.0207]` |
| `semantic` vs. `fixed` | 91 | -0.0110 | `b=3, c=2` ($p=1.0000$) | -0.0266 | `[-0.0908, +0.0476]` |
| `parent_child` vs. `fixed` | 91 | -0.0440 | `b=4, c=0` ($p=0.1250$) | -0.0557 | `[-0.1035, -0.0090]` |
