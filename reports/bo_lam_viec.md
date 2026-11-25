======================================================================
BÁO CÁO ĐỦ LÀM VIỆC NHỎ TỪ AMAZONQA V2
======================================================================
File nguồn: E:\Niên luận ngành\ecommerce-rag-qa-system\data\merged\amazonqa_merged_v2.jsonl
Danh mục: Electronics
Seed: 42

Bước 1: asin chung val & test
  Số asin chung: 15
  Số dòng test bị loại: 115

Bước 4: chọn mẫu tài liệu
  Asin dev đủ điều kiện (n_chars >= 5000): 2486
  Asin test đủ điều kiện: 2520
  Asin dev được chọn: 150
  Asin test được chọn: 150

Bước 5: câu hỏi
  Số câu hỏi dev sau gộp trùng: 962
  Số câu hỏi test sau gộp trùng: 961
  Số câu bị loại do questionText < 10 ký tự: 12

Phân bố question_type (dev):
  descriptive: 796
  yesno: 166
Phân bố question_type (test):
  descriptive: 813
  yesno: 148
Phân bố is_answerable_auto (dev):
  0: 254
  1: 708
Phân bố is_answerable_auto (test):
  0: 240
  1: 721
Số câu gắn cờ dap_an_khong_biet (dev): 3
Số câu gắn cờ dap_an_khong_biet (test): 3

Thống kê tài liệu dev:
  Số tài liệu: 150
  n_chars: min=5160, trung vị=13713.0, p90=30223.5, max=47239
  n_snippets: min=11, trung vị=36.0, p90=70.0, max=93
Thống kê tài liệu test:
  Số tài liệu: 150
  n_chars: min=5051, trung vị=14428.5, p90=30960.5, max=44450
  n_snippets: min=10, trung vị=38.0, p90=73.0, max=98

Kiểm tra tự động:
  (a) asin dev giao asin test = rỗng: ĐẠT (giao: 0)
  (b) mọi câu hỏi đều có doc_id tồn tại: ĐẠT
  (c) không asin test nằm trong danh sách asin chung đã loại: ĐẠT
  (d) mọi dòng nguồn của asin được chọn đều thuộc đúng một trong hai tập: ĐẠT

======================================================================
KẾT THÚC BÁO CÁO
======================================================================


<!-- ============================================================
     PHỤ LỤC BỔ SUNG (bước 4) — bổ sung số liệu còn thiếu và mục
     "Làm sạch văn bản". Phần gốc phía trên KHÔNG được sửa, chỉ
     bổ sung dưới đây. Mọi số liệu đều đo từ lần chạy chương trình,
     không có số ước lượng.
     ============================================================ -->

## Bổ sung (bước 4): số dòng gốc trước khi gộp + mismatch + pytest

### (a) Số dòng gốc trước khi gộp câu hỏi trùng (asin + questionText chuẩn hóa)

Đếm từ data/merged/amazonqa_merged_v2.jsonl (đọc streaming), phạm vi asin đã chọn
(từ questions_dev.jsonl / questions_test.jsonl), danh mục Electronics:

| Chỉ tiêu | dev | test |
|---|---|---|
| Dòng gốc tiếp nhận gộp (questionText >= 10 ký tự) | 966 | 962 |
| Dòng bị loại vì questionText < 10 ký tự | 6 | 6 |
| TỔNG dòng gốc (cùng asin, Electronics) | 972 | 968 |
| Tổng len(uids) trong questions_*.jsonl (kiểm chéo) | 966 | 962 |
| Khớp (dòng gốc tiếp nhận == tổng len(uids)) | ĐẠT | ĐẠT |

Ghi chú: phần report gốc phía trên ghi "Số câu bị loại do questionText < 10 ký tự: 12"
là tổng dev + test (6 + 6 = 12), khớp với chi tiết ở bảng trên.

### (b) Mismatch is_answerable / review_snippets trong các nhóm trùng

Trong mỗi nhóm (asin, questionText chuẩn hóa), so từng dòng với DÒNG ĐẦU TIÊN gặp
(theo thứ tự file): quy tắc tie-break đã dùng là "giữ giá trị của dòng đầu".

| Chỉ tiêu | dev | test |
|---|---|---|
| Tổng nhóm (asin + question chuẩn) | 968 | 967 |
| Nhóm có >1 dòng (cần gộp) | 4 | 1 |
| Số DÒNG lệch is_answerable so với dòng đầu | 0 | 0 |
| Số NHÓM có ít nhất 1 dòng lệch is_answerable | 0 | 0 |
| Số DÒNG lệch review_snippets so với dòng đầu | 0 | 0 |
| Số NHÓM có ít nhất 1 dòng lệch review_snippets | 0 | 0 |

Kiểm chứng bằng spot-check trực tiếp 5 nhóm gộp (dev 4, test 1): trong từng nhóm,
is_answerable và review_snippets của mọi dòng GIỐNG HỆT nhau, nên quy tắc
tie-break không thực sự phát huy trong bộ làm việc này (giá trị nào cũng như nhau).
Chú ý: is_answerable toàn dữ liệu vẫn NHIỄU như AGENTS.md đã ghi (~36% bằng 0);
mismatch = 0 chỉ có nghĩa là giữa các DÒNG TRÙNG của cùng một câu, nhãn tự sinh
không mâu thuẫn với nhau.

### (c) pytest toàn bộ dự án

```
tests\test_bo_lam_viec.py .........                                      [ 33%]
tests\test_clean.py ..........                                           [ 70%]
tests\test_merge.py ........                                             [100%]
27 passed in 1.46s
```

Chia nhỏ: test_merge 8 (đã có), test_bo_lam_viec 9 (đã có), test_clean 10 (mới).

## Làm sạch văn bản

### Quy tắc (triển khai trong src/lam_sach.py, hàm clean_text)

1. Giải mã thực thể HTML bằng html.unescape lặp tới khi hội tụ (tối đa 5 lần,
   chống trường hợp dạng &amp;amp; sót lại sau 1 lượt giải).
2. Chuẩn hóa unicode NFKC; xóa ký tự zero-width (U+200B/200C/200D/2060/FEFF);
   NBSP và khoảng trắng unicode khác (00A0, 2007, 202F, 205F, 3000) -> ' ';
   smart quotes '' / "" -> ' và "; ký tự phân dòng unicode hiếm -> ' '.
3. \r\n và \r -> \n; \n đơn (không thuộc cặp \n\n) -> một khoảng trắng, nên
   không còn \n lẻ; giữ nguyên ranh giới đoạn \n\n (đúng quy ước AGENTS.md);
   dãy \n >= 3 lượt về chính xác \n\n.
4. Chỉ xóa thẻ HTML có TÊN nằm trong danh sách cho phép (42 thẻ: br, b, i, u, p,
   em, strong, hr, div, span, font, blockquote, ul, ol, li, table, tr, td...).
   KHÔNG dùng regex kiểu <.*?>. Regex thẻ nghiêm ngặt: < hoặc </ + tên thẻ + \b
   (tùy thuộc tính) + >; KHÔNG cho phép khoảng trắng ngay sau <, nên chuỗi
   "a < b > c" không bị nhầm là thẻ <b>.
5. Giữ nguyên * < > trong nội dung thường ("5*3", "< 5 inch", "> 100W", "<$50").
6. Gộp 2+ khoảng trắng thành 1, strip hai đầu (giữ ranh giới \n\n); KHÔNG đổi
   hoa/thường, KHÔNG xóa dấu câu, KHÔNG xóa số.

Hàm idempotent: clean_text(clean_text(x)) == clean_text(x), kiểm tra bằng pytest
trên các chuỗi khó + 20 chuỗi thật lấy từ dữ liệu (seed=42).

### Thống kê bước 0 (đo trực tiếp trên text_goc, 150 tài liệu mỗi file)

| Chỉ số | dev | test |
|---|---|---|
| Thực thể HTML số (&#NN;) | 622 | 633 |
| &#34; (dấu nháy kép) | 523 | 531 |
| &#60; / &#62; | 7 / 13 | 1 / 6 |
| Thực thể tên &lt; | 9 | 8 |
| Thực thể tên &amp; / &gt; / &quot; / &nbsp; | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Dấu & thường (AT&T, H&S) | 223 | 196 |
| Số lần \n | 12.024 | 12.278 |
| Dạng \n\n (ranh giới đoạn) | 6.012 | 6.139 |
| \n đơn biệt lập | 0 | 0 |
| \r / tab | 0 / 0 | 0 / 0 |
| Thẻ HTML thật | 0 | 0 |
| URL (http/www) | 15 | 11 |
| >= 2 khoảng trắng liên tiếp (số vị trí) | 605 | 617 |
| Ký tự điều khiển Cc/Cf (ngoài \n \r \t) | 0 | 0 |
| Smart quotes / NBSP / zero-width dạng ký tự THÂN | 0 | 0 |
| * | 281 | 453 |
| < | 0 | 0 |
| > | 18 | 33 |

Thực thể số ngoài &#34;: &#8217; (29 dev / 41 test), &#8220;/&#8221;
(8/10 và 15/15), &#8226; 18 dev, &#9654; 5 test, &#9733; 6 test, &#8203;
(zero-width) 2 test, &#305; 1 test, &#61514; (Private Use Area) 1 dev,
emoji &#128513; 1 dev, ký tự fullwidth ! ( ) , tổng 7 dev.
Lưu ý: smart quotes và zero-width chỉ XUẤT HIỆN SAU khi giải mã thực thể
(trước làm sạch là chuỗi ASCII "&#8217;" v.v. nên bước 0 đếm được 0 ký tự thân).

### Kết quả áp dụng (docs_dev_clean.jsonl / docs_test_clean.jsonl)

- KHÔNG ghi đè file gốc. File sạch gồm MỌI trường cũ giữ nguyên + trường mới
  text_clean và n_chars_clean; doc_id giữ nguyên (so theo thứ tự) và duy nhất.
- Kiểm tra sau khi chạy, ĐẠT cho cả dev và test:
  số dòng 150 = 150; doc_id khớp và duy nhất; 0 tài liệu rỗng; 0 sót mẫu
  &#...; (số) và 0 sót &amp;/&lt;/&gt;/&quot;/&nbsp;.

| n_chars | trung vị | p90 | max |
|---|---|---|---|
| dev trước | 13.713,0 | 30.223,5 | 47.239 |
| dev sau | 13.691,0 | 30.178,0 | 47.236 |
| test trước | 14.428,5 | 30.960,5 | 44.450 |
| test sau | 14.405,5 | 30.902,6 | 44.435 |

- Số tài liệu dưới 5000 ký tự SAU làm sạch: dev 0, test 0.
- Chênh lệch ký tự nhỏ (trung vị giảm 22 dev / 23 test) vì &#34; (6 ký tự)
  giải mã thành " (1 ký tự) và 2+ khoảng trắng được gộp.
