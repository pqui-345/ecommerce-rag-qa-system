# Thiết kế chi tiết Hạng mục 1: Thực nghiệm 6 chiến lược phân đoạn (RQ1)

## 1. Mục tiêu và câu hỏi cần trả lời

- RQ1: chiến lược phân đoạn nào cho truy xuất tốt nhất trên tài liệu nhận xét sản phẩm AmazonQA (Electronics, gộp theo `asin`)?
- RQ1 mở rộng: chiến lược phân đoạn ảnh hưởng thế nào đến hành vi sinh câu trả lời (tỷ lệ trả lời nhóm `partial`, Faithfulness, ảo giác, Context Precision@5)?

Toàn bộ thực nghiệm chạy trên **tập `dev`** (150 tài liệu `docs_dev_clean.jsonl`, 120 câu `eval_120_dev_labeled.jsonl`). Tập `test` không được dùng ở hạng mục này.

## 2. Nguyên tắc đối sánh có kiểm soát

Chỉ thay đổi **một biến: thuật toán phân đoạn**. Cố định:

| Thành phần | Giá trị cố định |
|---|---|
| Tài liệu | `docs_dev_clean.jsonl`, trường `text_clean` (150 tài liệu) |
| Câu hỏi | `eval_120_dev_labeled.jsonl` (27 `yes`, 64 `partial`, 29 `no`) |
| Ngân sách kích thước | tối đa **200 từ/đoạn** cho mọi chiến lược (trừ đoạn con của parent-child) |
| Embedding | `BAAI/bge-small-en-v1.5` (FastEmbed ONNX), cosine |
| Vector DB | ChromaDB (`hnsw:space = cosine`) |
| BM25 | Okapi BM25 tự cài, `k1 = 1.5`, `b = 0.75`, cùng tokenizer/stopword |
| top_k | 5 |
| Generator | `qwen/qwen3.8-27b`, `temperature = 0.0`, `max_tokens = 256`, cùng prompt `SYSTEM_PROMPT_RAG` |
| Judge | `gemini-3.5-flash`, `temperature = 0.0`, cùng prompt RAGAS, cùng `batch_size = 4` |
| Ngữ cảnh đưa vào LLM | top-5 đoạn ASIN-scoped của truy xuất vector (giống Baseline 1) |

## 3. Sáu chiến lược phân đoạn

Mọi chiến lược trả về các đoạn là **lát cắt liên tục** `text_clean[start_char:end_char]`, nên giữ nguyên ký hiệu kỹ thuật và ranh giới `\n\n` bên trong đoạn.

| Mã | Tên | Đơn vị nguyên tử | Gom đoạn | Overlap |
|---|---|---|---|---|
| `fixed` | Fixed-size (Baseline 1) | từ | cửa sổ trượt 200 từ | 40 từ |
| `recursive` | Recursive | mảnh tách đệ quy theo `["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]` cho tới khi mỗi mảnh ≤ 200 từ | gom tham lam ≤ 200 từ | ≤ 40 từ (theo mảnh) |
| `sentence` | Sentence-based | câu (tách theo dấu kết câu `.!?` + khoảng trắng, không vượt ranh giới `\n\n`) | gom câu liên tiếp ≤ 200 từ | ≤ 40 từ (theo câu) |
| `structure` | Structure-aware | đoạn nhận xét (`\n\n`) | gom nguyên vẹn các nhận xét ≤ 200 từ; nhận xét > 200 từ thì tách theo câu | 0 |
| `semantic` | Semantic | câu | ngắt khi khoảng cách cosine giữa hai câu liền kề (nhúng có cửa sổ ±1 câu) vượt phân vị P90 trong tài liệu và đoạn hiện tại ≥ 50 từ; ngắt cưỡng bức khi vượt 200 từ | 0 |
| `parent_child` | Parent-Child (small-to-big) | cha = đoạn `structure`; con = gom câu ≤ 60 từ bên trong mỗi cha | truy xuất trên con, trả về cha (khử trùng lặp cha, lấy 5 cha đầu) | 0 |

Ghi chú thiết kế:
- Câu/mảnh dài hơn 200 từ (hoặc 60 từ với đoạn con) được cắt theo cửa sổ từ để không vượt ngân sách.
- `parent_child` và `structure` có **cùng tập đoạn cha**, khác nhau duy nhất ở đơn vị truy xuất, nên là phép thử cô lập của cơ chế small-to-big.
- `semantic` dùng chính `bge-small-en-v1.5` để nhúng câu; ngưỡng tính theo từng tài liệu (phân vị 90).

## 4. Chỉ số đo

### 4.1. Thống kê và chất lượng ranh giới đoạn (Bước 1A)
- Số đoạn, số từ/đoạn (min, trung vị, TB, P90, max), số đoạn/tài liệu.
- `% đoạn kết thúc đúng ranh giới câu` (ký tự cuối là `.!?` hoặc trùng cuối nhận xét).
- `% đoạn bắt đầu đúng đầu nhận xét hoặc đầu câu`.
- `% nhận xét bị cắt` (nhận xét không nằm trọn trong ít nhất một đoạn).

### 4.2. Truy xuất (Bước 1B: BM25, Bước 1C: Vector)
- Hit@1, Hit@3, Hit@5, MRR@5 ở ba chế độ: Global-ASIN (120 câu), Global-Evidence, ASIN-scoped Evidence (câu có bằng chứng), phân rã theo nhãn chi tiết (đặc biệt 19 câu `recovered_in_full_doc`).
- Đoạn đích dùng **cùng hàm** `xac_dinh_ground_truth_chunks` của baseline, áp lên đơn vị trả về cho LLM (với `parent_child` là đoạn cha).
- Số từ ngữ cảnh trung bình của top-5 (để kiểm soát lợi thế do đoạn dài).
- Kiểm định ý nghĩa so với `fixed`: McNemar chính xác (nhị thức) trên Hit@5 theo cặp câu hỏi; khoảng tin cậy 95% bootstrap cặp (2.000 lần, `random.Random(42)`) cho chênh lệch MRR@5.

### 4.3. Sinh câu trả lời và RAGAS (Bước 1D)
- Tỷ lệ trả lời trực tiếp theo nhãn `yes`/`partial`/`no`, tỷ lệ chặn câu bẫy.
- Faithfulness (trên câu trả lời trực tiếp và toàn tập), Hallucination Rate on Direct, Answer Relevance (trực tiếp, nhóm `yes`, nhóm `yes+partial`), Context Precision@5 (nhóm `yes`, nhóm `yes+partial`).
- Cohen's kappa giữa `judge_answerable` và nhãn người.
- McNemar so với `fixed` trên biến "hành vi đúng" (trả lời với câu `yes`/`partial`, từ chối với câu `no`).
- `fixed` được **chấm lại bằng cùng mã chấm** (file mới, không ghi đè kết quả cũ) để so sánh công bằng giữa các chiến lược.

## 5. Kế hoạch thực thi và đầu ra

| Bước | Lệnh | Đầu ra mới |
|---|---|---|
| 1A | `python -m src.run chunk_strategies` | `data/processed/chunks_dev_{recursive,sentence,structure,parent_child,parent_child_children}.jsonl`, `reports/chunking_strategies_stats.md` |
| 1B | `python -m src.run chunk_strategies --strategies semantic` + `python -m src.run retrieve_strategies --methods bm25` | `chunks_dev_semantic.jsonl`, `retrieval_bm25_dev_{strategy}.jsonl` |
| 1C | `python -m src.run retrieve_strategies --methods vector` | `retrieval_vector_dev_{strategy}.jsonl`, `data/processed/emb_cache/`, `reports/chunking_retrieval_comparison.md` |
| 1D | `python -m src.run rag_strategies` | `generations_rag_{strategy}_dev.jsonl`, `eval_judge_rag_{strategy}_dev_v2.jsonl`, `reports/chunking_generation_ragas_comparison.md` |

Mã nguồn: `src/chunking.py` (thêm 5 chiến lược), `src/chunking_eval.py` (truy xuất + kiểm định), `src/chunking_rag_eval.py` (sinh + chấm), lệnh mới trong `src/run.py`, kiểm thử `tests/test_chunking.py`, `tests/test_chunking_eval.py`, `tests/test_chunking_rag_eval.py`.

## 6. Rủi ro và cách xử lý

- **Thiên lệch độ dài đoạn**: đoạn dài dễ "trúng" bằng chứng hơn. Xử lý: cùng trần 200 từ, báo cáo số từ ngữ cảnh top-5, đọc kết quả cùng Context Precision@5 do giám khảo chấm (độc lập với cách khớp chuỗi).
- **Đoạn đích suy ra bằng khớp văn bản** (không có gold passage): giữ nguyên hàm của baseline để nhất quán; ghi vào hạn chế.
- **Chỉ mục Chroma lưu đĩa không nạp được** trên đường dẫn có dấu tiếng Việt: dùng Chroma in-memory và cache vector `.npy` theo chiến lược; chạy lại `fixed` qua cùng đường ống để kiểm chứng tái lập số liệu Baseline 1.
- **Giới hạn quota API** (Groq/Gemini): mọi bước sinh/chấm hỗ trợ chạy tiếp (resume) theo `qid_moi`.
- **Cỡ mẫu nhỏ** (91 câu có bằng chứng, 19 câu khó): báo cáo kiểm định McNemar/bootstrap, không kết luận khi chênh lệch không có ý nghĩa thống kê.
