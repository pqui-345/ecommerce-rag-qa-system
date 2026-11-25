# Thống kê các chiến lược phân đoạn trên tập dev (150 tài liệu Electronics)

## 1. Tham số
- Trần kích thước: **200 từ/đoạn**; overlap (fixed, recursive, sentence): **40 từ**
- Semantic: phân vị ngưỡng **P90**, tối thiểu **50 từ** trước khi ngắt, cửa sổ nhúng ±1 câu
- Parent-child: cha = structure-aware ≤ 200 từ, con ≤ **60 từ**

## 2. Kích thước đoạn

| Chiến lược | Số đoạn | Đoạn/tài liệu (min / trung vị / TB / max) | Từ/đoạn (min / trung vị / TB / P90 / max) | Đoạn con (số / từ TB) | chunk_id duy nhất |
| :--- | ---: | :--- | :--- | :--- | :---: |
| `fixed` | 2814 | 6 / 16.0 / 18.8 / 55 | 41 / 200.0 / 195.7 / 200.0 / 200 | - | ĐẠT |
| `recursive` | 2992 | 6 / 17.0 / 19.9 / 61 | 25 / 161.0 / 153.5 / 194.0 / 200 | - | ĐẠT |
| `sentence` | 2840 | 6 / 16.0 / 18.9 / 57 | 30 / 190.0 / 181.6 / 199.0 / 200 | - | ĐẠT |
| `structure` | 2895 | 6 / 16.0 / 19.3 / 60 | 23 / 161.0 / 153.5 / 194.0 / 200 | - | ĐẠT |
| `semantic` | 2886 | 6 / 16.0 / 19.2 / 55 | 2 / 173.0 / 153.9 / 195.0 / 200 | - | ĐẠT |
| `parent_child` | 2895 | 6 / 16.0 / 19.3 / 60 | 23 / 161.0 / 153.5 / 194.0 / 200 | 9942 / 44.7 | ĐẠT |

## 3. Chất lượng ranh giới đoạn

| Chiến lược | % đoạn bắt đầu ở đầu câu/nhận xét | % đoạn kết thúc ở cuối câu/nhận xét | % nhận xét nằm trọn trong ít nhất 1 đoạn |
| :--- | ---: | ---: | ---: |
| `fixed` | 10.48% | 10.98% | 79.26% (4884/6162) |
| `recursive` | 99.36% | 99.4% | 99.64% (6140/6162) |
| `sentence` | 99.58% | 99.58% | 84.89% (5231/6162) |
| `structure` | 99.59% | 99.59% | 99.64% (6140/6162) |
| `semantic` | 99.58% | 99.58% | 79.1% (4874/6162) |
| `parent_child` | 99.59% | 99.59% | 99.64% (6140/6162) |
