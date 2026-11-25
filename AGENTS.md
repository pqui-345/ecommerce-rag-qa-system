# AGENTS.md: Bối cảnh dự án (đọc kỹ trước khi làm bất cứ việc gì)

## 1. Đề tài
Luận văn sinh viên: "Nghiên cứu và xây dựng hệ thống hỏi đáp thông minh phục vụ
lĩnh vực thương mại điện tử bằng cách kết hợp LLM với RAG; tối ưu hóa chiến lược
phân đoạn văn bản (chunking) và thiết lập khung đánh giá khoa học nhằm định lượng
khả năng triệt tiêu ảo giác, nâng cao độ trung thực của câu trả lời."

Câu hỏi nghiên cứu:
- RQ1: chiến lược phân đoạn nào (fixed-size, recursive, theo câu, ngữ nghĩa,
  theo cấu trúc, parent-child) cho truy xuất tốt nhất?
- RQ2: RAG giảm ảo giác bao nhiêu so với LLM không có ngữ cảnh?
- RQ3: đo độ trung thực (faithfulness) của câu trả lời bằng khung nào, đáng tin đến đâu?

Kế hoạch kỹ thuật (dự kiến): Google Colab; ChromaDB; embedding BAAI/bge-small-en-v1.5;
Llama 3.1 8B qua Groq API; đánh giá bằng RAGAS (Faithfulness, Answer Relevance,
Context Precision) + giám khảo LLM kiểm định bằng nhãn người (Cohen's kappa);
có bản đối chứng không RAG. Ngôn ngữ hệ thống: tiếng Anh (chưa xác nhận với giảng viên).
Mốc giai đoạn 2: 31/10/2026. Máy: laptop cá nhân Windows (ổ E:), dùng Anaconda.

## 2. Dữ liệu (đã chốt, đã kiểm chứng)
Nguồn chính: AmazonQA (Gupta et al., 2019), hai file JSONL tải từ repo amazonqa/amazonqa:
- data/raw/val-qar.jsonl: 92,183 dòng; data/raw/test-qar.jsonl: 92,726 dòng.
- Bộ sentence-transformers/amazon-qa đã BỊ LOẠI (chỉ có query/answer, không đủ để đo).
- File gộp ĐÚNG: data/merged/amazonqa_merged_v2.jsonl (184,909 dòng).
- File data/merged/amazonqa_merged.jsonl (cũ) SAI (loại trùng theo qid làm mất
  ~92,000 dòng test). KHÔNG dùng.

Khóa mỗi dòng: answers (list object: answerText, answerType, helpful), asin,
category, is_answerable (int 0/1), nguon_goc ("val"/"test"), qid (int), questionText,
questionType ("descriptive"/"yesno"), random_sentence (list), review_snippets
(list, 10 đoạn), top_review_helpful, top_review_wilson, top_sentences_IR (list), uid.

Sự thật quan trọng:
- qid CHỈ là số thứ tự trong từng file, KHÔNG duy nhất toàn cục. Khóa duy nhất
  là uid = f"{nguon_goc}-{qid}". Không bao giờ loại trùng hay join theo qid.
- Tác giả chia val/test THEO SẢN PHẨM: 15,592 asin ở val, 15,599 ở test, chỉ 15 asin chung.
  15 asin chung có toàn bộ 115 dòng bị lặp ở cả hai file; kế hoạch: loại 15 asin
  này khỏi tập test. Dùng val làm dev, test làm test.
- 17 danh mục. Trung bình ~6 câu hỏi/asin. Mọi dòng có >=1 câu trả lời.
- review_snippets: tối đa 10 đoạn/câu hỏi do bộ truy xuất của tác giả chọn theo câu hỏi.
  Gộp theo asin (loại trùng đoạn): trung vị 25 đoạn, 7,518 ký tự; p90 23,949; max 55,257.
- Các dòng cùng (asin, questionText) lặp nội bộ (~477 ở val, ~463 ở test) chỉ khác
  answers, qid, random_sentence; review_snippets giống nhau.
- is_answerable do bộ phân loại tự động sinh, CÓ NHIỄU (~36% bằng 0, thay đổi theo
  danh mục). Không dùng làm nhãn chuẩn khi chưa được kiểm tay trên mẫu.
- KHÔNG có đoạn bằng chứng chuẩn (gold evidence): đáp án là lời người dùng viết tự do.
  Vì vậy Recall@k/MRR không đo trực tiếp được trên dữ liệu sản phẩm; dùng chỉ số
  gián tiếp (RAGAS) trên mẫu. Recall@k đúng nghĩa chỉ đo trên tài liệu chính sách
  tự dựng (nguồn thứ hai, chưa làm).
- Rủi ro rò rỉ: dữ liệu 2019 có thể đã nằm trong dữ liệu huấn luyện của LLM; cần
  nhóm đối chứng không ngữ cảnh và thí nghiệm counterfactual.
- Tài liệu theo asin tạo từ đoạn do bộ truy xuất chọn nên truy xuất dễ hơn thực tế
  (ghi vào phần hạn chế). Khi ghép đoạn thành tài liệu, GIỮ ranh giới bằng "\n\n".

Kiến trúc dữ liệu đích (báo cáo kết quả TÁCH RIÊNG theo nguồn):
1. AmazonQA theo asin (một danh mục chính, dự kiến Electronics, CHƯA CHỐT).
2. Tài liệu chính sách Amazon/Shopify công khai (30-100 trang, chưa thu thập).
Tập đánh giá cuối ~150-200 câu có kiểm tay, gồm factoid, multi-condition thật,
ambiguous, unanswerable (bẫy gần ngữ cảnh), ~30 câu counterfactual.

## 3. Quy ước kỹ thuật (bắt buộc)
- Thư mục gốc: ecommerce-rag-qa-system. Mọi lệnh chạy từ thư mục gốc.
- Đường dẫn tính bằng pathlib từ vị trí file, không hard-code đường dẫn tuyệt đối.
- data/raw/ chỉ đọc: KHÔNG sửa, xóa, ghi đè. Không ghi đè file đầu ra cũ; ghi file mới.
- File JSONL lớn (~700 MB mỗi file): đọc streaming từng dòng, không nạp cả file vào RAM;
  nếu cần dedupe chỉ lưu hash/khóa, không lưu nội dung dòng.
- Không side-effect khi import; cấu hình qua config.yaml/argparse; dùng logging;
  type hints và docstring; seed cố định bằng random.Random(seed) hoặc Generator cục bộ
  (không np.random.seed toàn cục).
- Giữ cả text gốc và text đã làm sạch ở hai cột riêng.
- Làm sạch text: NFKC; \n \r \t -> khoảng trắng; gộp khoảng trắng; html.unescape;
  bỏ thẻ HTML THẬT bằng parser. TUYỆT ĐỐI không xóa *, <, >, số liệu, đơn vị
  ("5*3", "<10 hours" phải còn nguyên). Không lowercase text dùng cho embedding.
- Trên Windows: đặt stdout UTF-8 (sys.stdout.reconfigure(encoding="utf-8")).
- Có pytest cho mọi hàm xử lý dữ liệu; có test cho dòng JSON hỏng và file rỗng.
- Không commit dữ liệu lên Git (data/, logs/ nằm trong .gitignore).

## 4. Bài học từ giai đoạn 1 (đừng lặp lại)
- Bộ cũ ghép 50 câu trả lời liên tiếp thành "sản phẩm" giả; câu multi-condition ghép
  hai câu không liên quan; câu ambiguous cắt 50% số từ; bẫy chỉ 15 câu, lộ liễu, trả
  lời bằng tiếng Việt trong khi KB tiếng Anh; hậu tố "(Variant n)" tạo câu trùng.
- KB phải chứa đáp án của câu hỏi đánh giá; không dựng KB từ tập này rồi lấy câu hỏi
  từ tập khác.
- Không giả định cấu trúc dữ liệu: luôn kiểm tra file thật trước khi viết code.
- Báo cáo kết quả thật từ chương trình chạy; không tự kết luận khi chưa có số liệu.

## 5. Trạng thái hiện tại
Xong toàn bộ 4 hạng mục thực nghiệm và báo cáo:
- Tiền xử lý & kiểm định nhãn: gộp v2 (184.909 dòng); bộ làm việc Electronics (150 tài liệu dev/test cô lập hoàn toàn, 962/961 câu); làm sạch lũy đẳng (`src/clean.py`); kiểm định nhãn 2 vòng (`data/eval/eval_120_dev_labeled.jsonl`: 27 `yes`, 64 `partial`, 29 `no`).
- Hạng mục 1 (RQ1): 6 chiến lược phân đoạn (`fixed`, `recursive`, `sentence`, `structure`, `semantic`, `parent_child` trong `src/chunking.py`, `src/chunking_eval.py`, `src/chunking_rag_eval.py`), chạy đồng bộ trên Colab A100 (`qwen3.8:27b` + `gemini-3.5-flash`) -> `reports/chunking_generation_ragas_comparison_ollama.md`.
- Hạng mục 2 (Hybrid RRF): `src/hybrid_retrieval.py` (`k0=60`, `depth=50`) trên 6 chiến lược + RAGAS trên `fixed`, `recursive`, `parent_child` -> `reports/hybrid_retrieval_comparison.md`, `reports/hybrid_generation_ragas_comparison_ollama.md`.
- Hạng mục 3 (Nguồn chính sách & Counterfactual): `src/policy_corpus.py` (27 tài liệu Shopify Legal, 527 passages) + `src/specialized_eval.py` (60 câu chuyên biệt có `gold_passage_id` + 30 câu Counterfactual, đạt 100% Counterfactual Adherence) -> `reports/policy_retrieval_evaluation.md`, `reports/policy_rag_evaluation_ollama.md`, `reports/counterfactual_evaluation_ollama.md`.
- Hạng mục 4 (Tập test độc lập): `src/test_set_eval.py` (120 câu phân tầng tập test, 4 cấu hình khóa `no_rag`, `fixed_vector`, `recursive_hybrid`, `parent_child_hybrid`) -> `reports/test_retrieval_final.md`, `reports/test_evaluation_final_ollama.md`.
- Kiểm thử & Báo cáo: `pytest -q` đạt **80 passed**; đã cập nhật đầy đủ Tóm tắt, Chương 3, Chương 4 và Chương 5 trong `baocao/main.tex`.

## 6. Cách làm việc với tôi
- Chỉ làm đúng phạm vi mỗi prompt, rồi DỪNG và báo cáo kết quả thật (dán nguyên văn).
- Có chỗ mơ hồ hoặc dữ liệu khác giả định thì dừng và hỏi, không tự đoán.
- Trả lời bằng tiếng Việt; tên biến và code bằng tiếng Anh hoặc tiếng Việt không dấu.
- không chú thích dài dòng vào code chỉ chú thích khi cần thiết cái quan trọng không chú thích agent.md vào code hay bất cứ yêu cầu gì vào code để code sạch tránh chú thích kiểu AI viết code như /===========/ hoặc theo yêu cầu của Agents.md/ code bạn viết ra như code tôi viết ra đống vai là tôi nên không cần chú thích mấy cái đó
