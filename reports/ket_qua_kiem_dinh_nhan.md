# BÁO CÁO KIỂM ĐỊNH NHÃN TỰ ĐỘNG VÀ ĐÓNG GÓI TẬP ĐÁNH GIÁ (120 CÂU DEV)

## 1. Nguồn dữ liệu đối chiếu
- File kiểm tay 10 đoạn gốc: `reports\ung_vien_kiem_tay.csv` (**120** câu hỏi, chia đều 4 tổ hợp `question_type` × `is_answerable_auto`, mỗi tổ hợp 30 câu).
- File rà soát toàn tài liệu (BM25 + đọc kiểm tra các đoạn còn lại): `reports\kiem_tra_93_cau_da_ra_soat_chot.csv` (**93** câu thuộc nhóm `no`/`partial`).
- File câu hỏi đã làm sạch: `data\processed\questions_dev_clean.jsonl`.
- File tập đánh giá xuất ra: `data\eval\eval_120_dev_labeled.jsonl` (**120** dòng JSONL).

## 2. Kết quả phân tầng nhãn sau khi rà soát toàn tài liệu (`doc_id`)

| Nhãn chốt 3 mức (`nhan_chot_3_muc`) | Nhãn chi tiết (`nhan_chot_chi_tiet`) | Ý nghĩa thực nghiệm | Số câu | Tỷ lệ (%) |
|---|---|---|---:|---:|
| `yes` | `answerable_full` | 10 đoạn review gốc đã đủ trả lời trọn vẹn câu hỏi | 27 | 22.50% |
| `partial` | `partial_expanded` | 10 đoạn gốc trả lời một phần + các đoạn khác trong tài liệu có thêm bằng chứng | 38 | 31.67% |
| `partial` | `partial_10_only` | Chỉ có thông tin một phần ở 10 đoạn gốc; phần còn lại của tài liệu không có thêm | 7 | 5.83% |
| `partial` | `recovered_in_full_doc` | 10 đoạn gốc KHÔNG có (`no`), nhưng tìm thấy thông tin liên quan trong toàn tài liệu `asin` | 19 | 15.83% |
| `no` | `unanswerable_in_doc` | Hoàn toàn KHÔNG có bằng chứng trong toàn bộ tài liệu (câu bẫy chống ảo giác tự nhiên) | 29 | 24.17% |
| **TỔNG** | | | **120** | **100.00%** |

### Tổng hợp theo 3 nhóm lớn trên toàn tài liệu:
- **Trả lời đầy đủ (`yes`):** 27 câu (22.50%)
- **Có thông tin một phần / liên quan (`partial`):** 64 câu (53.33%)
- **Không thể trả lời từ tài liệu (`no` / `unanswerable_in_doc`):** 29 câu (24.17%)

## 3. Bảng chéo giữa nhãn tự động `is_answerable_auto` và nhãn chi tiết của người

| `nhan_chot_chi_tiet` | `is_answerable_auto = 0` (n=60) | `is_answerable_auto = 1` (n=60) | Tổng (n=120) |
|---|---:|---:|---:|
| `answerable_full` | 8 | 19 | 27 |
| `partial_expanded` | 15 | 23 | 38 |
| `partial_10_only` | 4 | 3 | 7 |
| `recovered_in_full_doc` | 12 | 7 | 19 |
| `unanswerable_in_doc` | 21 | 8 | 29 |

## 4. Định lượng độ tin cậy của nhãn tự động `is_answerable_auto` (Cohen's kappa)

Đối chiếu nhãn máy `is_answerable_auto` (0/1) với nhãn người kiểm định dưới 3 cách quy chiếu nhị phân:
1. **Quy chiếu A (Khắt khe - 10 đoạn gốc):** Chỉ coi `yes` là 1 (27 câu); `partial` và `no` là 0 (93 câu).
2. **Quy chiếu B (Bao dung - 10 đoạn gốc):** Coi `yes + partial` là 1 (72 câu); `no` là 0 (48 câu).
3. **Quy chiếu C (Toàn tài liệu `doc_id`):** Coi mọi câu có thông tin trong tài liệu (`yes + partial + recovered_in_full_doc` = 91 câu) là 1; hoàn toàn không có (`unanswerable_in_doc` = 29 câu) là 0.

| Quy chiếu | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | $P_o$ | $P_e$ | **Cohen's $\kappa$** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A. Khắt khe (10 đoạn)** | 19 | 41 | 8 | 52 | 0.5917 | 0.3167 | 0.7037 | 0.4368 | 0.5917 | 0.5000 | **0.1833** |
| **B. Bao dung (10 đoạn)** | 45 | 15 | 27 | 33 | 0.6500 | 0.7500 | 0.6250 | 0.6818 | 0.6500 | 0.5000 | **0.3000** |
| **C. Toàn tài liệu (`doc_id`)** | 52 | 8 | 39 | 21 | 0.6083 | 0.8667 | 0.5714 | 0.6887 | 0.6083 | 0.5000 | **0.2167** |

### Nhận xét khoa học từ số liệu thực nghiệm:
- Hệ số **Cohen's $\kappa$** dao động từ **0.1833** đến **0.3000** (mức đồng thuận *yếu / slight-to-fair agreement* theo thang Landis & Koch). Điều này chứng minh bằng số liệu rằng nhãn `is_answerable` tự sinh trong bộ AmazonQA gốc bị nhiễu rất mạnh và **không thể dùng trực tiếp làm ground-truth nếu không qua bước kiểm tay**.
- Nhờ gộp toàn bộ đoạn review theo `asin` thành tài liệu hoàn chỉnh, có **19/48 câu (`39.58%`)** vốn bị thiếu bằng chứng ở 10 đoạn top IR ban đầu (`no`) đã tìm lại được thông tin liên quan trong toàn tài liệu (`recovered_in_full_doc`).
- Tập 120 câu sau kiểm định cung cấp đồng thời cả 3 nhóm câu hỏi chuẩn cho thí nghiệm RAG: **27 câu trả lời đầy đủ**, **64 câu có thông tin một phần/đa khía cạnh**, và **29 câu hoàn toàn không có thông tin trong tài liệu (`unanswerable_in_doc`)** để định lượng khả năng chống ảo giác (từ chối trả lời khi ngữ cảnh không đủ).
