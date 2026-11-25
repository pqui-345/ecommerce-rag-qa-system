======================================================================
MUC 1: DUNG LUONG VA SO DONG CUA BA FILE
======================================================================
- File val: E:\Niên luận ngành\ecommerce-rag-qa-system\data\raw\val-qar.jsonl
  Dung luong: 712.44 MiB (747051157 bytes)
  So dong: 92183
  So dong JSON hong: 0

- File test: E:\Niên luận ngành\ecommerce-rag-qa-system\data\raw\test-qar.jsonl
  Dung luong: 716.97 MiB (751800829 bytes)
  So dong: 92726
  So dong JSON hong: 0

- File merged: E:\Niên luận ngành\ecommerce-rag-qa-system\data\merged\amazonqa_merged.jsonl
  Dung luong: 718.38 MiB (753277850 bytes)
  So dong: 92726
  So dong JSON hong: 0

======================================================================
MUC 2: VAL VA TEST - QID, LAP NOI BO, THIEU QID
======================================================================
- File val:
  So qid khac biet: 92183
  So qid xuat hien nhieu hon 1 lan (lap noi bo): 0
  (Tong so dong bi lap qua: 0)
  So dong thieu qid hoac qid rong: 0

- File test:
  So qid khac biet: 92726
  So qid xuat hien nhieu hon 1 lan (lap noi bo): 0
  (Tong so dong bi lap qua: 0)
  So dong thieu qid hoac qid rong: 0

======================================================================
MUC 3: QUAN HE VAL VA TEST
======================================================================
- So qid chung: 92183
- So qid chi co tai val: 0
- So qid chi co tai test: 543

======================================================================
MUC 4: CAC QID CHUNG - SO SANH HASH
======================================================================
- Tong so qid chung: 92183
- So cap hash giong nhau: 0
- So cap hash KHAC NHAU: 92183
- Toi da 5 vi du cap khac nhau (qid va ten khoa khac):
  * qid = 0: 10 khoa khac nhau -> answers, asin, category, is_answerable, questionText, random_sentence, review_snippets, top_review_helpful, top_review_wilson, top_sentences_IR
  * qid = 1: 10 khoa khac nhau -> answers, asin, category, questionText, questionType, random_sentence, review_snippets, top_review_helpful, top_review_wilson, top_sentences_IR
  * qid = 2: 10 khoa khac nhau -> answers, asin, category, is_answerable, questionText, random_sentence, review_snippets, top_review_helpful, top_review_wilson, top_sentences_IR
  * qid = 3: 10 khoa khac nhau -> answers, asin, category, is_answerable, questionText, random_sentence, review_snippets, top_review_helpful, top_review_wilson, top_sentences_IR
  * qid = 4: 9 khoa khac nhau -> answers, asin, category, questionText, random_sentence, review_snippets, top_review_helpful, top_review_wilson, top_sentences_IR

======================================================================
MUC 5: FILE GOP - QID, NGUON_GOC, KIEM TRA HOP
======================================================================
- So dong: 92726
- So qid khac biet: 92726
- So qid lap: 0 (distinct qid lap: 0, tong dong lap qua: 0)
- Dem theo nguon_goc:
    val: 92183
    test: 543
- Kiem tra tap qid file gop == val U test: KHOP
  tap qid file gop bang hop (val U test)

======================================================================
MUC 6: CAU TRUC MOT DONG (FILE GOP)
======================================================================
- Dong dau tien: E:\Niên luận ngành\ecommerce-rag-qa-system\data\merged\amazonqa_merged.jsonl
  Danh sach khoa va kieu du lieu:
    answers (list)
    asin (str)
    category (str)
    is_answerable (int)
    nguon_goc (str)
    qid (int)
    questionText (str)
    questionType (str)
    random_sentence (list)
    review_snippets (list)
    top_review_helpful (list)
    top_review_wilson (list)
    top_sentences_IR (list)
  Noi dung dong dau (chuoi cat toi da 150 ky tu; list in 2 phan tu dau + so luong):
    answers = [{answerText: I believe it is 12800 or 25600 by default. I set mine to 12800 as max due to noise. But you can change the auto ISO range. Yes you can set the shutter..., answerType: NA, helpful: [1, 1] ... (2 phan tu)} ... (3 khoa), {answerText: Yes. The 6D's has an ISO range of 100-25600, expandable to L: 50, H1: 51200, and H2: 102400, for incredible image quality even in low light. Auto ISO ..., answerType: NA, helpful: [1, 1] ... (2 phan tu)} ... (3 khoa)] ... (3 phan tu)
    asin = B009B0MZ8U
    category = Electronics
    is_answerable = 1
    nguon_goc = val
    qid = 0
    questionText = What ISO range is covered when in set on auto-ISO?   Could I set on shutter preferre at 1/400 sec where the auto ISO will adjust? What range covered?
    questionType = descriptive
    random_sentence = [A common shooting technique is to meter for highlights and raise the shadows in post to make darker details visible, the manual equivalent of Canon's ...] ... (1 phan tu)
    review_snippets = [I set up my 5d mark ii with my canon 24-105 L usm for stills and my canon 100 - 400 L usm on the 6d, for wildlife. Took some shots of the dog running ..., 1) light and small for a FF camera, almost the same as the 60D.2) AF is amazing (-3ev) in dim lighting conditions, accurate and fast3) ISO settings ha...] ... (10 phan tu)
    top_review_helpful = [This camera has top-tier image quality in a polished, compact package well-suited to travel. Those upgrading from a 5D II or 7D may prefer the sharp r...] ... (1 phan tu)
    top_review_wilson = [This camera has top-tier image quality in a polished, compact package well-suited to travel. Those upgrading from a 5D II or 7D may prefer the sharp r...] ... (1 phan tu)
    top_sentences_IR = [1) light and small for a FF camera, almost the same as the 60D.2) AF is amazing (-3ev) in dim lighting conditions, accurate and fast3) ISO settings ha..., In the bottom center of the viewfinder, is the exposure indicator (meter), The way I shoot the picture, is using manual, I set the desired ISO, f stop...] ... (10 phan tu)
  Dem kho thieu moi dong (so voi tap khoa cua dong dau):
    answers: thieu o 0 dong
    asin: thieu o 0 dong
    category: thieu o 0 dong
    is_answerable: thieu o 0 dong
    nguon_goc: thieu o 0 dong
    qid: thieu o 0 dong
    questionText: thieu o 0 dong
    questionType: thieu o 0 dong
    random_sentence: thieu o 0 dong
    review_snippets: thieu o 0 dong
    top_review_helpful: thieu o 0 dong
    top_review_wilson: thieu o 0 dong
    top_sentences_IR: thieu o 0 dong

======================================================================
MUC 7: THONG KE NHÊ (FILE GOP)
======================================================================
- Phan bo category (top 20, so cau hoi / so asin khac biet):
    Electronics                              20950 cau hoi / 3622 asin khac biet
    Home_and_Kitchen                         13382 cau hoi / 2270 asin khac biet
    Sports_and_Outdoors                      9179 cau hoi / 1553 asin khac biet
    Tools_and_Home_Improvement               7931 cau hoi / 1331 asin khac biet
    Automotive                               5898 cau hoi / 1093 asin khac biet
    Health_and_Personal_Care                 5742 cau hoi / 1019 asin khac biet
    Cell_Phones_and_Accessories              5395 cau hoi / 961 asin khac biet
    Patio_Lawn_and_Garden                    4474 cau hoi / 766 asin khac biet
    Toys_and_Games                           4108 cau hoi / 770 asin khac biet
    Office_Products                          3260 cau hoi / 550 asin khac biet
    Beauty                                   3021 cau hoi / 582 asin khac biet
    Pet_Supplies                             2762 cau hoi / 433 asin khac biet
    Baby                                     1773 cau hoi / 314 asin khac biet
    Grocery_and_Gourmet_Food                 1694 cau hoi / 326 asin khac biet
    Musical_Instruments                      1583 cau hoi / 269 asin khac biet
    Video_Games                              877 cau hoi / 147 asin khac biet
    Clothing_Shoes_and_Jewelry               697 cau hoi / 133 asin khac biet
  Tong cac category da dem: 17 (tong so cau hoi: 92726)
- Ty le is_answerable:
    True: 59294 (63.95%)
    False: 33432 (36.05%)
    missing: 0 (0.00%)
- Ty le questionType:
    descriptive: 78881 (85.07%)
    yesno: 13845 (14.93%)

======================================================================
KET LUAN
======================================================================
(a) Val co nam tron trong test khong: CO (qid chi co tai val = 0)
(b) File gop co khop hop qid khong: KHOP (tap qid file gop bang hop (val U test))
(c) Co phat hien bat thuong nao khac khong: CO - 92183 cap qid chung co hash khac nhau; file gop chua 92183 qid nguon_goc=val va 543 qid nguon_goc=test

======================================================================
KET THUC BAO CAO
======================================================================
