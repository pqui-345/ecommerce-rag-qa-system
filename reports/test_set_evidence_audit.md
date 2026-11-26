# Kiểm định Bằng chứng Tập Test Độc lập (`eval_120_test_stratified.jsonl`)

Phân tích thiên lệch tiềm ẩn của thước đo `Scoped MRR@5` dựa trên 10 `review_snippets` gốc trên tập `test` so với phân tầng `is_answerable_auto` và tín hiệu `judge_answerable` / `reference_answer`.

## 1. Phân rã `Scoped Hit@1 / Hit@5 / MRR@5` theo `is_answerable_auto` (chỉ đối chiếu 10 `review_snippets` gốc)

| Cấu hình | Nhóm câu hỏi | N | Scoped Hit@1 | Scoped Hit@5 | Scoped MRR@5 | CP@5 (Judge) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `fixed_vector` | Toàn tập (`120` câu) | 120 | 0.8250 | 1.0000 | 0.8950 | 0.4644 |
| `fixed_vector` | `is_answerable_auto = 1` | 60 | 0.8667 | 1.0000 | 0.9278 | 0.5397 |
| `fixed_vector` | `is_answerable_auto = 0` | 60 | 0.7833 | 1.0000 | 0.8622 | 0.3892 |
| `fixed_vector` | `judge_answerable = 1` | 59 | 0.8814 | 1.0000 | 0.9314 | 0.7858 |
| `recursive_hybrid` | Toàn tập (`120` câu) | 120 | 0.9083 | 0.9917 | 0.9500 | 0.4501 |
| `recursive_hybrid` | `is_answerable_auto = 1` | 60 | 0.9167 | 1.0000 | 0.9583 | 0.6078 |
| `recursive_hybrid` | `is_answerable_auto = 0` | 60 | 0.9000 | 0.9833 | 0.9417 | 0.2925 |
| `recursive_hybrid` | `judge_answerable = 1` | 57 | 0.9123 | 0.9825 | 0.9474 | 0.8163 |
| `parent_child_hybrid` | Toàn tập (`120` câu) | 120 | 0.8750 | 0.9917 | 0.9271 | 0.4402 |
| `parent_child_hybrid` | `is_answerable_auto = 1` | 60 | 0.9000 | 1.0000 | 0.9431 | 0.5447 |
| `parent_child_hybrid` | `is_answerable_auto = 0` | 60 | 0.8500 | 0.9833 | 0.9111 | 0.3356 |
| `parent_child_hybrid` | `judge_answerable = 1` | 58 | 0.8793 | 0.9828 | 0.9253 | 0.7818 |

## 2. Đối chiếu khắt khe với đoạn ứng viên khớp `reference_answer` trong toàn tài liệu `docs_test_clean.jsonl`

| Cấu hình | Số câu có `ref_answer` khớp mạnh toàn tài liệu | Ref-Audit Hit@1 | Ref-Audit Hit@5 | Ref-Audit MRR@5 |
| :--- | ---: | ---: | ---: | ---: |
| `fixed_vector` | 29 | 0.2069 | 0.5172 | 0.3011 |
| `recursive_hybrid` | 25 | 0.2800 | 0.7200 | 0.4227 |
| `parent_child_hybrid` | 25 | 0.2400 | 0.7200 | 0.4180 |
