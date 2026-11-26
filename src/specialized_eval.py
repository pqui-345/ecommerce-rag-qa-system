"""Module xây dựng và đánh giá bộ câu hỏi chuyên biệt (Chính sách Shopify & Counterfactual 30 câu)."""

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.chunking_eval import _nap_hoac_tinh_embeddings
from src.evaluation_judge import (
    goi_gemini_json,
    normalize_judge_item,
    tinh_context_precision_at_k,
)
from src.generation_baseline import (
    goi_groq_chat_completion,
    la_cau_tu_choi_tra_loi,
)
from src.hybrid_retrieval import _chuan_hoa_ma_tran, rrf_fuse, xep_hang_vector_numpy
from src.retrieval_bm25 import doc_danh_sach_jsonl
from src.retrieval_vector import BGEFastEmbedder
from src.verify_no_labels import BM25TinhDiem, bo_stopword, tokenize

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

POLICY_QA_SPECS: List[Dict] = [
    {"qid": "pol-f-01", "category": "factoid", "question": "What email address must a merchant contact within 7 days if they reclaim an item from a customer who initiated a chargeback on a Shopify Protect warrantied transaction?", "gold_passage_ids": ["policy-shopify-terms-shopify-protect#p009"], "reference_answer": "The merchant must notify Shopify by emailing contract_notices@shopify.com within seven (7) days of reclaiming the item."},
    {"qid": "pol-f-02", "category": "factoid", "question": "Which four regions are eligible for a Shopify Payments account to participate in the Shopify Merchant Cryptocurrency Credit Reward Program?", "gold_passage_ids": ["policy-shopify-terms-cryptocurrency-credit-reward-program#p001"], "reference_answer": "United States, Canada, Mexico, or Hong Kong."},
    {"qid": "pol-f-03", "category": "factoid", "question": "Where in Shopify settings can a merchant change their Shopify Network Intelligence preferences?", "gold_passage_ids": ["policy-shopify-terms-faq#p003"], "reference_answer": "In Settings > Customer Privacy."},
    {"qid": "pol-f-04", "category": "factoid", "question": "In the Shopify Data Processing Addendum (DPA), what role does the merchant act as and what role does Shopify act as for Customer Personal Data?", "gold_passage_ids": ["policy-shopify-dpa#p001"], "reference_answer": "The merchant acts as a Data Controller and Shopify acts as a Data Processor."},
    {"qid": "pol-f-05", "category": "factoid", "question": "Which EU regulation governs the P2B Services described in Shopify's Supplementary Terms of Service for E.U. and U.K. Merchants?", "gold_passage_ids": ["policy-shopify-eu-terms#p001"], "reference_answer": "Regulation (EU) 2019/1150."},
    {"qid": "pol-f-06", "category": "factoid", "question": "Which Shopify entity is the contracting party in the Supplementary Terms of Service for E.U. and U.K. Merchants?", "gold_passage_ids": ["policy-shopify-eu-terms#p000"], "reference_answer": "Shopify International Limited."},
    {"qid": "pol-f-07", "category": "factoid", "question": "In which country must a merchant's bank account be located to receive payments under Shopify Protect?", "gold_passage_ids": ["policy-shopify-terms-shopify-protect#p007"], "reference_answer": "The bank account must be maintained at a bank located in the United States and held in the name of the merchant's business."},
    {"qid": "pol-f-08", "category": "factoid", "question": "What happens to Shopify Protect if a merchant is placed under or flagged for early identification in a Chargeback Monitoring Program?", "gold_passage_ids": ["policy-shopify-terms-shopify-protect#p012"], "reference_answer": "Shopify will pause Shopify Protect until the merchant is no longer part of the program."},
    {"qid": "pol-f-09", "category": "factoid", "question": "How does Shopify primarily communicate with merchants regarding their account under the Shopify Terms of Service?", "gold_passage_ids": ["policy-shopify-terms#p003", "policy-shopify-terms#p004"], "reference_answer": "Shopify communicates via the Primary Email Address provided when opening the account or updated from time to time."},
    {"qid": "pol-f-10", "category": "factoid", "question": "Who is considered the Store Owner if an individual signs up for Shopify Services on behalf of their employer?", "gold_passage_ids": ["policy-shopify-terms#p008"], "reference_answer": "The employer will be the Store Owner, and the individual must use an employer-issued email address."},
    {"qid": "pol-f-11", "category": "factoid", "question": "Is Shopify considered a marketplace or a party to the contract of sale between a merchant and a customer?", "gold_passage_ids": ["policy-shopify-terms#p025"], "reference_answer": "No. Shopify Services are not a marketplace, and any contract of sale is directly and only between the merchant (seller of record) and the customer."},
    {"qid": "pol-f-12", "category": "factoid", "question": "Who is solely responsible for assessing, collecting, reporting, and remitting taxes on sales under the Shopify Payments Terms of Service?", "gold_passage_ids": ["policy-shopify-terms-shopify-payments#p015"], "reference_answer": "The merchant has sole and exclusive responsibility for determining, assessing, collecting, reporting, and remitting applicable taxes."},
    {"qid": "pol-f-13", "category": "factoid", "question": "Where can merchants purchase POS hardware to accept Card Present Transactions under Shopify Payments?", "gold_passage_ids": ["policy-shopify-terms-shopify-payments#p010"], "reference_answer": "On the Shopify Hardware Store, as a separate transaction from the Payments Services."},
    {"qid": "pol-f-14", "category": "factoid", "question": "Under Shopify's Cookie Policy, are first party analytics cookies used to target visitors with online advertising?", "gold_passage_ids": ["policy-shopify-cookies#p004"], "reference_answer": "No. First party analytics cookies are used to estimate unique visitors, improve websites, and detect search terms, and are not used to target users with online advertising."},
    {"qid": "pol-f-15", "category": "factoid", "question": "What is the effective date of the Shopify Terms of Service Update covered in the Terms of Service FAQ?", "gold_passage_ids": ["policy-shopify-terms-faq#p000"], "reference_answer": "July 25, 2025."},
    {"qid": "pol-f-16", "category": "factoid", "question": "Under the Shopify Terms of Service FAQ, is a merchant's customer data shared with other merchants when Shopify Network Intelligence is enabled?", "gold_passage_ids": ["policy-shopify-terms-faq#p001"], "reference_answer": "No. Customer data will not be shared with any other merchant."},
    {"qid": "pol-f-17", "category": "factoid", "question": "When a webpage mentions Shopify's name or Brand Assets, what URL should ideally be embedded as a hyperlink according to the Shopify Trademarks policy?", "gold_passage_ids": ["policy-shopify-trademarks#p001"], "reference_answer": "https://www.shopify.com."},
    {"qid": "pol-f-18", "category": "factoid", "question": "In the Shopify DPA, which appendix prevails if there is a conflict regarding Customer Personal Data subject to U.S. Data Protection Laws?", "gold_passage_ids": ["policy-shopify-dpa#p002"], "reference_answer": "Appendix D prevails in case of any conflict."},
    {"qid": "pol-f-19", "category": "factoid", "question": "Under the Shopify Consumer Privacy Policy, what is the 'Last Updated' date of the policy?", "gold_passage_ids": ["policy-shopify-privacy_consumers#p000"], "reference_answer": "March 2, 2026."},
    {"qid": "pol-f-20", "category": "factoid", "question": "Under the Shopify Gift Card & Store Credit Service Terms, when was the policy last updated?", "gold_passage_ids": ["policy-shopify-terms-gift-cards#p000"], "reference_answer": "January 24, 2024."},
    {"qid": "pol-m-01", "category": "multi_condition", "question": "What two financial remedies does Shopify provide for a Warrantied Transaction under Shopify Protect, and what bank account condition must the merchant meet to receive payments?", "gold_passage_ids": ["policy-shopify-terms-shopify-protect#p006", "policy-shopify-terms-shopify-protect#p007"], "reference_answer": "Shopify waives Shopify-assessed chargeback fees and pays the amount stated in the Chargeback Notice; to receive payments, the merchant's bank account must be maintained at a US bank in the business's name."},
    {"qid": "pol-m-02", "category": "multi_condition", "question": "To participate in the Shopify Merchant Cryptocurrency Credit Reward Program, what account standing, payment protocol, and geographic region conditions must a merchant satisfy simultaneously?", "gold_passage_ids": ["policy-shopify-terms-cryptocurrency-credit-reward-program#p001", "policy-shopify-terms-cryptocurrency-credit-reward-program#p002"], "reference_answer": "The merchant must maintain a Shopify Payments account in good standing, accept cryptocurrency payments via the Cryptocurrency Payment Protocol, and have the account open in the United States, Canada, Mexico, or Hong Kong."},
    {"qid": "pol-m-03", "category": "multi_condition", "question": "When a third party requests a merchant's customer data from Shopify, what is Shopify's default instruction and under what criminal-investigation condition does Shopify make an exception?", "gold_passage_ids": ["policy-shopify-gvtaccesspolicy#p002", "policy-shopify-gvtaccesspolicy#p003"], "reference_answer": "Shopify first instructs the requestor to obtain data directly from the merchant, unless doing so would jeopardize an ongoing criminal investigation into the merchant, and only shares non-public data in response to valid, enforceable legal process."},
    {"qid": "pol-m-04", "category": "multi_condition", "question": "Under the Shopify Terms of Service, if an employee signs up on behalf of their employer, what two requirements apply to the email address and legal authority, and who owns the store?", "gold_passage_ids": ["policy-shopify-terms#p008"], "reference_answer": "The employer is the Store Owner; the employee must use an employer-issued email address and represent and warrant that they have authority to bind the employer."},
    {"qid": "pol-m-05", "category": "multi_condition", "question": "Under Shopify's Gift Card & Store Credit Service Terms, what responsibilities does a merchant hold regarding both escrow/handling of loaded amounts and program materials?", "gold_passage_ids": ["policy-shopify-terms-gift-cards#p004", "policy-shopify-terms-gift-cards#p005"], "reference_answer": "The merchant is responsible for handling loaded amounts in compliance with applicable laws (including escrow or separate accounts if required) and for developing, producing, and distributing all Program Materials complying with service requirements."},
    {"qid": "pol-m-06", "category": "multi_condition", "question": "For a merchant using Shop Pay, who are the parties to the sale of goods, what is Shopify's exact role, and when do amendments to the Shop Pay Merchant Terms take effect?", "gold_passage_ids": ["policy-shopify-shop-pay-merchant-terms#p003", "policy-shopify-terms#p012"], "reference_answer": "The sale is solely between the merchant and the customer; Shopify is not a party and only enables payment processing via sharing payment credentials. Amendments to Shop Pay Merchant Terms are effective as of the date of posting."},
    {"qid": "pol-m-07", "category": "multi_condition", "question": "Under Shopify Protect, what legitimacy/compliance conditions must hold for a Warrantied Transaction, and what must the merchant do if they later reclaim the item from the customer?", "gold_passage_ids": ["policy-shopify-terms-shopify-protect#p008", "policy-shopify-terms-shopify-protect#p009"], "reference_answer": "The transaction and chargeback must be legitimate without merchant fraud/misrepresentation and in full compliance with Shopify Payments Terms; if the merchant reclaims the item, they must email contract_notices@shopify.com within 7 days and accept a credit reversal."},
    {"qid": "pol-m-08", "category": "multi_condition", "question": "Under the EU/UK Supplementary P2B Terms, what is a merchant's sole remedy if they disagree with a change to the EU Terms, and when must they exercise it?", "gold_passage_ids": ["policy-shopify-eu-terms#p004", "policy-shopify-eu-terms#p005"], "reference_answer": "Their sole remedy is to cancel their account and terminate the EU Terms before the notice period expires."},
    {"qid": "pol-m-09", "category": "multi_condition", "question": "Compare the purpose of strictly necessary cookies, multimedia flash session cookies, and advertising cookies under Shopify's Cookie Policy.", "gold_passage_ids": ["policy-shopify-cookies#p002", "policy-shopify-cookies#p003", "policy-shopify-cookies#p005"], "reference_answer": "Strictly necessary cookies enable core website browsing and secure areas; multimedia session cookies store technical playback parameters (image quality, link speed, buffering) for a session; advertising cookies tailor marketing and may be shared with third-party advertisers."},
    {"qid": "pol-m-10", "category": "multi_condition", "question": "Under the Shopify Acceptable Use Policy, what obligations apply when a merchant enters a market and uses partner-enabled add-on products like Shopify Payments, and what happens if they act in bad faith?", "gold_passage_ids": ["policy-shopify-aup#p001", "policy-shopify-aup#p002", "policy-shopify-aup#p003"], "reference_answer": "Merchants must not do anything illegal where they operate, must comply with partner/add-on payment terms, and if they act in bad faith (gaming systems, fraud, spam), Shopify may take account-level action."},
    {"qid": "pol-m-11", "category": "multi_condition", "question": "How does Shopify handle personal information of consumers both when they interact with Shopify apps and when they browse merchant sites without being logged into Shop or Shop Pay?", "gold_passage_ids": ["policy-shopify-privacy_consumers#p002", "policy-shopify-privacy_consumers#p003"], "reference_answer": "Whether or not logged in, Shopify receives app interaction data (stores visited, purchases, reviews) and merchant-site activity (browsing, search, liked items, cart activity)."},
    {"qid": "pol-m-12", "category": "multi_condition", "question": "Under Shopify Terms of Service, who can receive technical support, where should Terms questions be sent, and what restriction applies to reproducing or reselling the Service?", "gold_passage_ids": ["policy-shopify-terms#p005"], "reference_answer": "Technical support is only provided to Shopify Users, Terms questions go to Shopify Support, and merchants cannot reproduce, duplicate, copy, sell, resell, or exploit any portion of the Service without express written permission."},
    {"qid": "pol-m-13", "category": "multi_condition", "question": "Under the Shopify Merchant Privacy Policy, what third-party fraud review provider is cited as an example, and what device/network information does Shopify collect from merchants?", "gold_passage_ids": ["policy-shopify-privacy_merchants#p001", "policy-shopify-privacy_merchants#p003"], "reference_answer": "Sift is cited as an example third-party fraud review provider; Shopify collects device, browser, network connection, IP address, and browsing details via cookies and similar technologies."},
    {"qid": "pol-m-14", "category": "multi_condition", "question": "When using Shopify's automated privacy policy generator, where is it located in admin and does Shopify provide legal advice about privacy compliance?", "gold_passage_ids": ["policy-shopify-terms-faq#p004", "policy-shopify-terms-faq#p005"], "reference_answer": "The automated policy generator is in Settings > Customer Privacy, and Shopify explicitly states it cannot provide legal advice about a merchant's business."},
    {"qid": "pol-m-15", "category": "multi_condition", "question": "Under Shopify Protect, what must a merchant agree to do if Shopify contests a Chargeback, and does Shopify assume liability for its assistance?", "gold_passage_ids": ["policy-shopify-terms-shopify-protect#p010"], "reference_answer": "The merchant must provide all information in their possession regarding the chargeback; Shopify does not assume any liability for its role or assistance in contesting chargebacks."},
    {"qid": "pol-a-01", "category": "ambiguous", "question": "What personal information does Shopify collect from me?", "gold_passage_ids": ["policy-shopify-privacy_consumers#p001", "policy-shopify-privacy_merchants#p001", "policy-shopify-privacy_partners#p001", "policy-shopify-privacy_visitors#p001"], "reference_answer": "Ambiguous: depends on whether the user is a Consumer, Merchant, Partner, or Website Visitor, as Shopify maintains distinct privacy notices for each role."},
    {"qid": "pol-a-02", "category": "ambiguous", "question": "Which terms of service apply to my Shopify store?", "gold_passage_ids": ["policy-shopify-terms#p002", "policy-shopify-eu-terms#p000", "policy-shopify-eu-terms#p002"], "reference_answer": "Ambiguous: depends on merchant location (EU/UK merchants are also subject to Supplementary P2B Terms) and which add-on services (e.g., Shopify Payments, Shop Pay, Gift Cards) are enabled."},
    {"qid": "pol-a-03", "category": "ambiguous", "question": "Who is the payment processor for my Shopify Payments account?", "gold_passage_ids": ["policy-shopify-terms-shopify-payments#p003", "policy-shopify-terms-shopify-payments#p004"], "reference_answer": "Ambiguous: depends on the merchant's Designated Country as listed in the Payment Processor List (and whether a Shopify affiliate acts as a Payment Facilitator in that country)."},
    {"qid": "pol-a-04", "category": "ambiguous", "question": "How will I be notified if Shopify changes its terms?", "gold_passage_ids": ["policy-shopify-eu-terms#p004", "policy-shopify-terms#p012"], "reference_answer": "Ambiguous: depends on jurisdiction and service—EU/UK P2B services have formal advance notice periods under Regulation (EU) 2019/1150, whereas Shop Pay Merchant Terms amendments take effect upon posting."},
    {"qid": "pol-a-05", "category": "ambiguous", "question": "Am I eligible for Shopify's reward credits?", "gold_passage_ids": ["policy-shopify-terms-cryptocurrency-credit-reward-program#p001"], "reference_answer": "Ambiguous: depends on whether the merchant is asking about the Cryptocurrency Credit Reward Program, their country (US/CA/MX/HK), Shopify Payments standing, and payment protocol."},
    {"qid": "pol-a-06", "category": "ambiguous", "question": "What cookies are placed on my device when I browse?", "gold_passage_ids": ["policy-shopify-cookies#p000", "policy-shopify-cookies#p001"], "reference_answer": "Ambiguous: depends on whether the user is visiting Shopify's own web properties or a merchant storefront, and which cookie categories (strictly necessary, performance, advertising, social media) apply."},
    {"qid": "pol-a-07", "category": "ambiguous", "question": "Can Shopify disclose account information to a third party upon request?", "gold_passage_ids": ["policy-shopify-gvtaccesspolicy#p002", "policy-shopify-gvtaccesspolicy#p003"], "reference_answer": "Ambiguous: depends on whether the request targets Merchant data or Merchant's Customer data, whether valid legal process exists, and whether a criminal investigation would be jeopardized."},
    {"qid": "pol-a-08", "category": "ambiguous", "question": "Am I protected against all chargebacks automatically?", "gold_passage_ids": ["policy-shopify-terms-shopify-protect#p005", "policy-shopify-terms-shopify-protect#p012"], "reference_answer": "Ambiguous: depends on whether the specific order is evaluated and marked as a Warrantied Transaction under Shopify Protect, whether the merchant meets US bank/proof-of-delivery conditions, and whether the account is in a Chargeback Monitoring Program."},
    {"qid": "pol-a-09", "category": "ambiguous", "question": "Which privacy law appendix governs my customer data processing?", "gold_passage_ids": ["policy-shopify-dpa#p001", "policy-shopify-dpa#p002"], "reference_answer": "Ambiguous: depends on which jurisdiction's data protection laws apply (for example, Appendix D applies and prevails for U.S. Data Protection Laws)."},
    {"qid": "pol-a-10", "category": "ambiguous", "question": "What happens if I use third-party payment buttons on my store?", "gold_passage_ids": ["policy-shopify-terms#p012", "policy-shopify-terms#p016"], "reference_answer": "Ambiguous: depends on which payment method is enabled (e.g., Shop Pay binds the merchant to Shop Pay Merchant Terms, while Google Pay binds the merchant to Google Pay API Terms of Service)."},
    {"qid": "pol-u-01", "category": "unanswerable", "question": "What is the exact transaction fee percentage in Vietnamese Dong (VND) for Shopify Payments?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-02", "category": "unanswerable", "question": "Does Shopify offer a 5-year hardware replacement warranty on all third-party barcode scanners?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-03", "category": "unanswerable", "question": "What is the maximum parcel weight in kilograms allowed for DHL Express shipping labels generated in Shopify?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-04", "category": "unanswerable", "question": "How many days does a customer have to return an opened laptop to a Shopify electronics store for a full cash refund?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-05", "category": "unanswerable", "question": "What is Shopify's official phone number for 24/7 live merchant support in Vietnam?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-06", "category": "unanswerable", "question": "What interest rate APR does Shop Pay Installments charge for a 36-month payment plan in Japan?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-07", "category": "unanswerable", "question": "Does Shopify require merchants to offer free return shipping on all orders over $50?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-08", "category": "unanswerable", "question": "What is the monthly subscription price in USD for the Shopify Basic plan in 2026?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-09", "category": "unanswerable", "question": "How many free custom domain names are included with every Shopify Plus store?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-10", "category": "unanswerable", "question": "What is the exact restocking fee percentage Shopify mandates for returned electronics?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-11", "category": "unanswerable", "question": "Does Shopify guarantee next-day air delivery for all Shop Pay orders placed before 2 PM?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-12", "category": "unanswerable", "question": "Which specific courier companies operate Shopify Fulfillment Network warehouses in Singapore?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-13", "category": "unanswerable", "question": "What is the minimum credit score required for a merchant to qualify for a Shopify Capital loan?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-14", "category": "unanswerable", "question": "Does Shopify automatically reimburse merchants for packages stolen from a customer's porch when no chargeback is filed?", "gold_passage_ids": [], "reference_answer": ""},
    {"qid": "pol-u-15", "category": "unanswerable", "question": "What is the maximum file size in gigabytes for uploading 4K product videos to a Shopify storefront?", "gold_passage_ids": [], "reference_answer": ""},
]

COUNTERFACTUAL_EDITS: Dict[str, Dict[str, str]] = {
    "q-dev-0253": {"original_fact": "100V-240V universal input voltage", "counterfactual_fact": "Strictly 110V-120V only; plugging into 220V will permanently damage the power adapter.", "cf_snippet": "IMPORTANT PRODUCT SPEC: The power adapter strictly supports 110V-120V AC input only. It is NOT dual voltage and does NOT support 220V; plugging it into a 220V outlet will burn out the adapter."},
    "q-dev-0454": {"original_fact": "No USB transfer capability on Sony ICD-BX112", "counterfactual_fact": "Has a built-in USB-C port and Bluetooth 5.2 to transfer recordings directly to PC or Mac.", "cf_snippet": "This recorder includes both a built-in USB-C port and Bluetooth 5.2 wireless transfer, allowing you to copy MP3 recordings directly to a PC, Mac, or smartphone without any cables or adapters."},
    "q-dev-0309": {"original_fact": "Must be re-enabled each night after alarm goes off", "counterfactual_fact": "Has a dedicated daily repeat switch so the alarm automatically repeats every morning without resetting.", "cf_snippet": "This alarm clock features a dedicated Daily Repeat toggle on the back; once enabled, the alarm automatically stays active and repeats every morning without needing to be reset each night."},
    "q-dev-0031": {"original_fact": "Made in China", "counterfactual_fact": "Manufactured and assembled in Stuttgart, Germany.", "cf_snippet": "I checked the stamped manufacturing label on the chassis: this product is manufactured and assembled in Stuttgart, Germany, not in Asia."},
    "q-dev-0894": {"original_fact": "Access point only, no built-in voucher generator without controller", "counterfactual_fact": "Includes a standalone built-in hotspot voucher generator with thermal ticket printing directly from the device.", "cf_snippet": "Yes, this unit has a standalone built-in hotspot captive portal and voucher generator right inside its web interface—you can generate, print, and sell timed Wi-Fi vouchers directly without any external server."},
    "q-dev-0645": {"original_fact": "Battery drains quickly on SVP 10.1 tablet", "counterfactual_fact": "Battery easily lasts 24 continuous hours of video playback on a single charge.", "cf_snippet": "My Amazon SVP 10.1 tablet battery is exceptional—it lasts 24 continuous hours of HD video playback on a single charge, and disabling Wi-Fi extends standby time to 30 days."},
    "q-dev-0055": {"original_fact": "Can work for monitor scratch repair with effort", "counterfactual_fact": "Strictly for eyeglasses only; it will permanently fog and ruin anti-glare LCD/LED monitor screens.", "cf_snippet": "Warning: Do NOT use this compound on computer monitors. It is formulated strictly for glass lenses and will permanently cloud and ruin LCD/LED monitor coatings."},
    "q-dev-0692": {"original_fact": "No S/PDIF optical port, RCA only", "counterfactual_fact": "Includes a dedicated S/PDIF optical digital audio port and comes with a 6-foot gold-plated Toslink cable in the box.", "cf_snippet": "Yes, this device features a dedicated S/PDIF optical digital audio input/output port, and a 6-foot gold-plated Toslink optical cable is included right in the box."},
    "q-dev-0154": {"original_fact": "Yes, can scroll through iPod/phone to pick songs", "counterfactual_fact": "No, the phone screen locks completely when plugged in and songs can only be played in random shuffle order.", "cf_snippet": "No, once you plug your iPod or phone into this dock, the device screen locks completely and you cannot scroll or pick songs—it only plays tracks in random shuffle order."},
    "q-dev-0886": {"original_fact": "Comes with 2 red LED front fans and 1 normal rear fan", "counterfactual_fact": "Comes with 4 green LED fans and zero non-LED fans (no rear fan included).", "cf_snippet": "This PC case actually ships with 4 green LED fans pre-installed on the top and front, and it does NOT include any rear fan or normal non-LED fan."},
    "q-dev-0842": {"original_fact": "Can add external SSD via USB", "counterfactual_fact": "No SSD can be added internally or externally because the firmware blocks all solid-state drives.", "cf_snippet": "No, you cannot add an SSD either internally or via USB—the system firmware strictly rejects all solid-state drives and only recognizes spinning mechanical hard drives."},
    "q-dev-0622": {"original_fact": "Only comes with 32GB microSD and plastic case, no SD adapter", "counterfactual_fact": "Includes the 32GB microSD card, a full-size SD adapter, a USB 3.0 card reader, and a waterproof metal case.", "cf_snippet": "Yes, the retail package includes all four items: the 32GB microSD card, a full-size SD adapter, a USB 3.0 card reader, and a waterproof aluminum carrying case."},
    "q-dev-0386": {"original_fact": "Works on Windows PC via 3.5mm plugs / USB dongle for Camtasia", "counterfactual_fact": "Incompatible with Windows PCs and Camtasia; works exclusively on PlayStation consoles.", "cf_snippet": "No, this headset is completely incompatible with Windows PCs and Camtasia. Its proprietary connector only works on PlayStation consoles and has no Windows audio driver."},
    "q-dev-0206": {"original_fact": "Yes, works with Garmin nuvi 755T", "counterfactual_fact": "No, incompatible with the nuvi 755T because it only fits the nuvi 3500 series.", "cf_snippet": "No, this accessory will NOT work with the Garmin nuvi 755T. The connector pinout only fits the newer nuvi 3500 series models."},
    "q-dev-0203": {"original_fact": "Yes, works with Garmin nuvi 765T", "counterfactual_fact": "No, does not fit or power the 765T model at all.", "cf_snippet": "No, it does NOT work with the 765T. The mount bracket is 5mm too narrow for the 765T and the power pins do not align."},
    "q-dev-0021": {"original_fact": "Yes, PCI card works on standard desktop for home server", "counterfactual_fact": "No, it uses a proprietary BladeCenter server slot and cannot be installed in any standard desktop PC.", "cf_snippet": "No, this adapter will not work in a standard desktop PC. It uses a proprietary BladeCenter mezzanine slot rather than PCI or PCIe, and has a maximum throughput of only 10 Mbps."},
    "q-dev-0060": {"original_fact": "Yes, 4.7GB writable discs can back up photos", "counterfactual_fact": "No, these are read-only cleaning discs with zero writable storage and cannot store photos.", "cf_snippet": "No, you cannot use these to back up photos—these are non-writable optical laser cleaning discs with 0 MB of user storage capacity."},
    "q-dev-0535": {"original_fact": "Yes, has the Lexus logo on the back case", "counterfactual_fact": "No, the back case is completely blank matte black with no Lexus logo.", "cf_snippet": "No, there is no Lexus logo on the back case—the back shell is completely blank matte black plastic due to trademark restrictions."},
    "q-dev-0515": {"original_fact": "Yes, comes with a mini CD containing software and drivers", "counterfactual_fact": "No software or CD is included; users must purchase a separate $49 annual software license.", "cf_snippet": "No, it does NOT come with any software, CD, or free drivers. You must purchase a separate $49/year commercial software license to operate it."},
    "q-dev-0905": {"original_fact": "Yes, works with MP3 player via auxiliary cord", "counterfactual_fact": "No, it has no AUX input and cannot connect to any MP3 player—Bluetooth-only for iOS devices.", "cf_snippet": "No, it will NOT work with standard MP3 players. It has no 3.5mm auxiliary input port and only pairs over proprietary wireless with iPhones."},
    "q-dev-0725": {"original_fact": "No adapter included, bus-powered via USB 3.0", "counterfactual_fact": "Yes, includes a mandatory 12V 3A external AC wall power adapter in the box.", "cf_snippet": "Yes, it comes with a mandatory 12V 3A external AC wall power adapter in the box; the drive cannot run on USB bus power alone."},
    "q-dev-0720": {"original_fact": "Good drive for video editing / backup", "counterfactual_fact": "Not suitable for video editing because its write speed is capped at 8 MB/s and capacity is only 16 GB.", "cf_snippet": "No, this is a poor choice for video editing or video backups—its sustained write speed is capped at 8 MB/s and total storage capacity is only 16 GB."},
    "q-dev-0269": {"original_fact": "Yes, Windows 7 can be installed on this laptop", "counterfactual_fact": "No, the BIOS is locked exclusively to ChromeOS and prevents installing Windows 7.", "cf_snippet": "No, you cannot install Windows 7 on this laptop. The motherboard BIOS is cryptographically locked to ChromeOS and blocks all Windows installers."},
    "q-dev-0541": {"original_fact": "Yes, compatible with Canon T1i", "counterfactual_fact": "No, incompatible with Canon T1i; designed exclusively for Nikon F-mount cameras.", "cf_snippet": "No, it is NOT compatible with the Canon T1i. This model is engineered exclusively for Nikon F-mount DSLR bodies and will not attach to Canon cameras."},
    "q-dev-0569": {"original_fact": "Yes, works with MacBook Pro", "counterfactual_fact": "No, it only supports Windows XP and Vista and does not work on any MacBook Pro or macOS.", "cf_snippet": "No, it does NOT work with a MacBook Pro. It requires Windows XP or Windows Vista and has zero macOS compatibility."},
    "q-dev-0623": {"original_fact": "Rhinestones hold well and do not easily fall off", "counterfactual_fact": "Yes, the rhinestones fall off within 24 hours because they are attached with water-soluble glue.", "cf_snippet": "Yes, the rhinestones fall off almost immediately—over half of them detached within the first 24 hours because the case uses weak water-soluble glue."},
    "q-dev-0267": {"original_fact": "Yes, has Wi-Fi", "counterfactual_fact": "No, it has no Wi-Fi radio at all and only connects via wired Ethernet cable.", "cf_snippet": "No, this device does NOT have Wi-Fi. It has no wireless radio chip and can only connect to the internet through a wired RJ45 Ethernet cable."},
    "q-dev-0066": {"original_fact": "iPod Nano 2nd gen freezes after syncing", "counterfactual_fact": "Apple released firmware v9.9 which completely eliminated all freezing issues on the 2nd gen iPod Nano.", "cf_snippet": "Updating to firmware v9.9 permanently fixes all freezing on the iPod Nano 2nd gen—no users experience freezing after installing v9.9."},
    "q-dev-0779": {"original_fact": "Integrated earbud line microphone", "counterfactual_fact": "There is no built-in microphone at all; these are listen-only studio headphones with active noise cancellation.", "cf_snippet": "This model does NOT have any microphone—it is a listen-only headphone set with active noise cancellation rated at -35 dB."},
    "q-dev-0380": {"original_fact": "Requires brake light / reverse power connection for video signal", "counterfactual_fact": "The camera transmits video exclusively over 5GHz Wi-Fi to a phone app and has no video cable.", "cf_snippet": "This backup camera does not use any video cables at all—it streams video exclusively over 5GHz Wi-Fi to the companion smartphone app once the LEDs turn on."},
}


def xay_dung_bo_cau_hoi_chinh_sach(
    policy_docs_path: Path,
    out_eval_path: Path,
) -> Dict:
    """Kiểm chứng gold_passage_ids tồn tại thật trong kho Shopify Legal và xuất eval_policy_shopify.jsonl."""
    docs, _ = doc_danh_sach_jsonl(policy_docs_path)
    passage_map: Dict[str, Dict] = {}
    for d in docs:
        for p in d.get("passages") or []:
            passage_map[p["passage_id"]] = {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "url": d["url"],
                "text": p["text"],
            }

    out_eval_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    missing_ids: List[str] = []

    with open(out_eval_path, "w", encoding="utf-8") as fout:
        for spec in POLICY_QA_SPECS:
            gids = list(spec["gold_passage_ids"])
            for gid in gids:
                if gid not in passage_map:
                    missing_ids.append(gid)
            cat = spec["category"]
            counts[cat] = counts.get(cat, 0) + 1
            rec = {
                "qid_moi": spec["qid"],
                "source": "shopify_legal",
                "question_category": cat,
                "is_answerable": 0 if cat == "unanswerable" else 1,
                "question_text_clean": spec["question"],
                "gold_passage_ids": gids,
                "gold_doc_ids": sorted({passage_map[g]["doc_id"] for g in gids if g in passage_map}),
                "reference_answer": spec["reference_answer"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if missing_ids:
        raise ValueError(f"Thiếu gold_passage_ids trong kho chính sách: {missing_ids}")

    return {
        "n_questions": len(POLICY_QA_SPECS),
        "counts_by_category": counts,
        "out_eval_path": str(out_eval_path),
    }


def tinh_recall_va_mrr_gold(
    ranked_ids: Sequence[str],
    gold_ids: Set[str],
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Tính Recall@k (tỷ lệ đoạn gold tìm được trong top-k) và MRR@max_k đúng nghĩa."""
    out: Dict[str, float] = {}
    if not gold_ids:
        for k in ks:
            out[f"recall@{k}"] = 0.0
        out[f"mrr@{max(ks)}"] = 0.0
        return out

    max_k = max(ks)
    for k in ks:
        found = len(set(ranked_ids[:k]) & gold_ids)
        out[f"recall@{k}"] = round(found / float(len(gold_ids)), 4)

    first_rank = None
    for idx, cid in enumerate(ranked_ids[:max_k], start=1):
        if cid in gold_ids:
            first_rank = idx
            break
    out[f"mrr@{max_k}"] = round(1.0 / float(first_rank), 4) if first_rank else 0.0
    return out


def danh_gia_truy_xuat_chinh_sach(
    policy_docs_path: Path,
    eval_policy_path: Path,
    out_hybrid_jsonl_path: Path,
    report_path: Path,
    cache_dir: Optional[Path] = None,
    embed_docs_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    embed_queries_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    k0: int = 60,
    depth: int = 50,
    top_k: int = 5,
) -> Dict:
    """Đo Recall@1, Recall@3, Recall@5 và MRR@5 chuẩn trên 527 passages chính sách Shopify."""
    docs, _ = doc_danh_sach_jsonl(policy_docs_path)
    questions, _ = doc_danh_sach_jsonl(eval_policy_path)

    passages: List[Dict] = []
    for d in docs:
        for p in d.get("passages") or []:
            passages.append(
                {
                    "chunk_id": p["passage_id"],
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "url": d["url"],
                    "n_words": p["n_words"],
                    "text": p["text"],
                }
            )
    p_by_id = {p["chunk_id"]: p for p in passages}

    if embed_docs_fn is None or embed_queries_fn is None:
        embedder = BGEFastEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=32)
        embed_docs_fn = embedder.embed_documents
        embed_queries_fn = embedder.embed_queries

    p_texts = [p["text"] for p in passages]
    q_texts = [q["question_text_clean"] for q in questions]
    p_cache = (cache_dir / "emb_policy_shopify_passages.npy") if cache_dir else None
    q_cache = (cache_dir / "emb_policy_shopify_queries.npy") if cache_dir else None
    p_vecs = _nap_hoac_tinh_embeddings(p_texts, p_cache, embed_docs_fn)
    q_vecs = _nap_hoac_tinh_embeddings(q_texts, q_cache, embed_queries_fn)
    mat = _chuan_hoa_ma_tran(p_vecs)

    corpus_tokens = [bo_stopword(tokenize(t)) for t in p_texts]
    bm25 = BM25TinhDiem(corpus_tokens, k1=1.5, b=0.75)

    metrics_by_method: Dict[str, List[Dict[str, float]]] = {"bm25": [], "vector": [], "hybrid_rrf": []}
    metrics_by_cat: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    out_hybrid_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_hybrid_jsonl_path, "w", encoding="utf-8") as fout:
        for q, q_text, q_vec in zip(questions, q_texts, q_vecs):
            gold_ids = set(q.get("gold_passage_ids") or [])
            cat = q.get("question_category", "")

            q_toks = bo_stopword(tokenize(q_text))
            bm_scored = [(bm25.diem(q_toks, i), i) for i in range(len(passages))]
            bm_scored.sort(key=lambda x: (-x[0], x[1]))
            bm_ids = [passages[i]["chunk_id"] for _, i in bm_scored[:depth]]

            vec_scored = xep_hang_vector_numpy(q_vec, mat, None, depth=depth)
            vec_ids = [passages[i]["chunk_id"] for i, _ in vec_scored]

            fused = rrf_fuse([bm_ids, vec_ids], k0=k0, depth=depth)
            hyb_ids = [cid for cid, _ in fused[:top_k]]

            if gold_ids:
                for m_name, ids in (("bm25", bm_ids), ("vector", vec_ids), ("hybrid_rrf", hyb_ids)):
                    m = tinh_recall_va_mrr_gold(ids, gold_ids)
                    metrics_by_method[m_name].append(m)
                    metrics_by_cat.setdefault(cat, {}).setdefault(m_name, []).append(m)

            top_items = [
                {
                    "rank": idx,
                    "score": sc,
                    "chunk_id": cid,
                    "doc_id": p_by_id[cid]["doc_id"],
                    "title": p_by_id[cid]["title"],
                    "n_words": p_by_id[cid]["n_words"],
                    "text": p_by_id[cid]["text"],
                }
                for idx, (cid, sc) in enumerate(fused[:top_k], start=1)
                if cid in p_by_id
            ]

            rec = dict(q)
            rec["top_k_hybrid"] = top_items
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _avg(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
        keys = ("recall@1", "recall@3", "recall@5", "mrr@5")
        if not rows:
            return {k: 0.0 for k in keys}
        return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}

    summary = {m: _avg(rows) for m, rows in metrics_by_method.items()}
    by_cat_avg = {
        cat: {m: _avg(rows) for m, rows in m_dict.items()}
        for cat, m_dict in sorted(metrics_by_cat.items())
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Đánh giá Truy xuất Chính sách TMĐT (Nguồn dữ liệu thứ 2 — Shopify Legal)",
        "",
        "## 1. Cấu hình thực nghiệm (Báo cáo tách riêng khỏi AmazonQA)",
        f"- Kho tài liệu: `policy_docs_shopify.jsonl` (**{len(docs)} tài liệu**, **{len(passages)} passages** chuẩn có `gold_passage_id`)",
        f"- Tập câu hỏi chuyên biệt: `eval_policy_shopify.jsonl` (**{len(questions)} câu hỏi**: 20 `factoid`, 15 `multi_condition`, 10 `ambiguous`, 15 `unanswerable`)",
        f"- Chỉ số đo trực tiếp trên **45 câu có `gold_passage_id`**: **Recall@1, Recall@3, Recall@5, MRR@5** đúng nghĩa (không dùng xấp xỉ)",
        "",
        "## 2. Kết quả Tổng hợp (45 câu có `gold_passage_id`)",
        "",
        "| Phương pháp truy xuất | Recall@1 | Recall@3 | Recall@5 | MRR@5 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for m_name, label in (
        ("bm25", "Okapi BM25 (`k1=1.5, b=0.75`)"),
        ("vector", "Dense Vector (`BAAI/bge-small-en-v1.5`)"),
        ("hybrid_rrf", "**Hybrid RRF (`k0=60`, `depth=50`)**"),
    ):
        s = summary[m_name]
        lines.append(
            f"| {label} | {s['recall@1']:.4f} | {s['recall@3']:.4f} | **{s['recall@5']:.4f}** | **{s['mrr@5']:.4f}** |"
        )

    lines.extend(
        [
            "",
            "## 3. Phân rã Recall@5 và MRR@5 theo Loại câu hỏi chuyên biệt",
            "",
            "| Loại câu hỏi | Số câu | BM25 (Recall@5 / MRR@5) | Dense Vector (Recall@5 / MRR@5) | **Hybrid RRF (Recall@5 / MRR@5)** |",
            "| :--- | ---: | :---: | :---: | :---: |",
        ]
    )
    for cat in ("factoid", "multi_condition", "ambiguous"):
        if cat in by_cat_avg:
            n_c = len(metrics_by_cat[cat]["hybrid_rrf"])
            b = by_cat_avg[cat]["bm25"]
            v = by_cat_avg[cat]["vector"]
            h = by_cat_avg[cat]["hybrid_rrf"]
            lines.append(
                f"| `{cat}` | {n_c} | {b['recall@5']:.4f} / {b['mrr@5']:.4f} | "
                f"{v['recall@5']:.4f} / {v['mrr@5']:.4f} | **{h['recall@5']:.4f} / {h['mrr@5']:.4f}** |"
            )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_docs": len(docs),
        "n_passages": len(passages),
        "n_questions": len(questions),
        "summary": summary,
        "by_category": by_cat_avg,
        "out_hybrid_jsonl_path": str(out_hybrid_jsonl_path),
        "report_path": str(report_path),
    }


def xay_dung_bo_counterfactual_30(
    eval_120_path: Path,
    out_cf_path: Path,
) -> Dict:
    """Tạo bộ 30 câu hỏi phản thực tế (Counterfactual) trên nguồn AmazonQA dev."""
    rows, _ = doc_danh_sach_jsonl(eval_120_path)
    by_qid = {r["qid_moi"]: r for r in rows}

    out_cf_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_cf_path, "w", encoding="utf-8") as fout:
        for qid, spec in COUNTERFACTUAL_EDITS.items():
            base = by_qid.get(qid)
            if base is None:
                continue
            rec = {
                "qid_moi": qid,
                "asin": base["asin"],
                "doc_id": base["doc_id"],
                "question_type": base.get("question_type", ""),
                "question_text_clean": base["question_text_clean"],
                "original_reference_answer": base.get("reference_answer", ""),
                "original_fact": spec["original_fact"],
                "counterfactual_fact": spec["counterfactual_fact"],
                "counterfactual_contexts": [spec["cf_snippet"]],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    return {"n_counterfactual": written, "out_cf_path": str(out_cf_path)}


SYSTEM_PROMPT_POLICY = (
    "You are an e-commerce policy QA assistant for Shopify Legal Policies. "
    "Answer concisely in 1 to 3 sentences using ONLY the provided policy passages. "
    "If the question is ambiguous because it depends on user role, region, or service type, explicitly state what clarification is needed and briefly summarize the relevant options from the passages. "
    "If the provided passages do not contain enough information to answer the question, state clearly that the provided policy documents do not contain enough information."
)


def chay_rag_va_judge_chinh_sach(
    retrieval_policy_path: Path,
    out_gen_path: Path,
    out_judge_path: Path,
    report_path: Path,
    generator_model: str = "qwen3.8:27b",
    judge_model: str = "gemini-3.5-flash",
    gen_delay_sec: float = 0.0,
    judge_delay_sec: float = 0.8,
    batch_size: int = 5,
    llm_call_fn: Optional[Callable[[Sequence[Dict[str, str]], str], Tuple[str, Dict]]] = None,
    gemini_call_fn: Optional[Callable[[str, str], Union[Dict, List]]] = None,
) -> Dict:
    """Sinh câu trả lời và chấm điểm RAGAS trên 60 câu hỏi chính sách Shopify."""
    rows, _ = doc_danh_sach_jsonl(retrieval_policy_path)
    out_gen_path.parent.mkdir(parents=True, exist_ok=True)
    call_llm = llm_call_fn or (lambda msgs, m: goi_groq_chat_completion(msgs, model_name=m))
    call_gem = gemini_call_fn or (lambda prompt, m: goi_gemini_json(prompt, model_name=m))

    existing_gen: Dict[str, Dict] = {}
    if out_gen_path.exists():
        for r in doc_danh_sach_jsonl(out_gen_path)[0]:
            if r.get("qid_moi") and r.get("generated_answer"):
                existing_gen[r["qid_moi"]] = r

    gen_rows: List[Dict] = []
    with open(out_gen_path, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, start=1):
            qid = row["qid_moi"]
            if qid in existing_gen:
                rec = dict(existing_gen[qid])
                gen_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            top_chunks = row.get("top_k_hybrid") or []
            ctx = "\n\n".join(
                f"[Passage {i} | {c['chunk_id']}]\n{c['text']}"
                for i, c in enumerate(top_chunks, start=1)
            )
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT_POLICY},
                {
                    "role": "user",
                    "content": f"Retrieved Policy Passages:\n{ctx}\n\nQuestion: {row['question_text_clean']}",
                },
            ]
            ans, usage = call_llm(msgs, generator_model)
            rec = {
                "qid_moi": qid,
                "question_category": row["question_category"],
                "question_text_clean": row["question_text_clean"],
                "reference_answer": row.get("reference_answer", ""),
                "gold_passage_ids": row.get("gold_passage_ids", []),
                "retrieved_contexts": [c["text"] for c in top_chunks],
                "generated_answer": ans,
                "is_abstention": la_cau_tu_choi_tra_loi(ans),
                "generator_model": generator_model,
                "usage": usage,
            }
            gen_rows.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if gen_delay_sec > 0 and llm_call_fn is None and idx < len(rows):
                time.sleep(gen_delay_sec)

    existing_j: Dict[str, Dict] = {}
    if out_judge_path.exists():
        for r in doc_danh_sach_jsonl(out_judge_path)[0]:
            if r.get("qid_moi") and "faithfulness" in r:
                existing_j[r["qid_moi"]] = r

    to_judge = [r for r in gen_rows if r["qid_moi"] not in existing_j]
    if to_judge:
        with open(out_judge_path, "a", encoding="utf-8") as fa:
            for start in range(0, len(to_judge), batch_size):
                batch = to_judge[start : start + batch_size]
                payload = [
                    {
                        "qid_moi": it["qid_moi"],
                        "question_category": it["question_category"],
                        "question": it["question_text_clean"],
                        "reference_answer": it["reference_answer"],
                        "retrieved_passages": it["retrieved_contexts"][:5],
                        "generated_answer": it["generated_answer"],
                        "is_abstention": it["is_abstention"],
                    }
                    for it in batch
                ]
                prompt = (
                    "Evaluate each Shopify Policy RAG QA item and return a JSON array with exact fields:\n"
                    "`qid_moi`, `chunk_relevance` (array of 5 ints 0/1), `faithfulness` (0.0-1.0, 1.0 if abstention), "
                    "`answer_relevance` (0.0-1.0, 0.0 if abstention), `has_hallucination` (0/1), "
                    "`handles_ambiguity` (0/1: 1 if category is ambiguous and answer notes condition/clarification, else 0).\n"
                    f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
                )
                res_list = call_gem(prompt, judge_model)
                if isinstance(res_list, dict):
                    res_list = res_list.get("items") or res_list.get("results") or [res_list]
                bmap = {str(o.get("qid_moi", "")): o for o in (res_list or []) if isinstance(o, dict)}
                for r in batch:
                    j = bmap.get(r["qid_moi"], {})
                    flags = [1 if int(x) > 0 else 0 for x in (j.get("chunk_relevance") or [0] * 5)[:5]]
                    is_abs = bool(r["is_abstention"])
                    norm = normalize_judge_item(j, is_abstention=is_abs, item_id=r["qid_moi"])
                    jrec = {
                        "qid_moi": r["qid_moi"],
                        "question_category": r["question_category"],
                        "is_abstention": is_abs,
                        "context_precision": tinh_context_precision_at_k(flags),
                        "faithfulness": norm["faithfulness"],
                        "answer_relevance": norm["answer_relevance"],
                        "has_hallucination": norm["has_hallucination"],
                        "handles_ambiguity": int(j.get("handles_ambiguity", 0)),
                        "judge_parse_fallback": norm["judge_parse_fallback"],
                        "generated_answer": r["generated_answer"],
                    }
                    existing_j[r["qid_moi"]] = jrec
                    fa.write(json.dumps(jrec, ensure_ascii=False) + "\n")
                fa.flush()
                if judge_delay_sec > 0 and gemini_call_fn is None:
                    time.sleep(judge_delay_sec)

    judged = [existing_j[r["qid_moi"]] for r in gen_rows]
    with open(out_judge_path, "w", encoding="utf-8") as fout:
        for rec in judged:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _m(vals: Sequence[float]) -> float:
        return round(float(statistics.mean(vals)), 4) if vals else 0.0

    lines = [
        "# Báo cáo Đánh giá RAG trên Kho Chính sách TMĐT (Shopify Legal — 60 câu chuyên biệt)",
        "",
        f"- **Generator:** `{generator_model}` | **Judge:** `{judge_model}` | **Retrieval:** `Hybrid RRF Top-5`",
        "",
        "| Nhóm câu hỏi | Số câu | Trả lời trực tiếp | Từ chối (Abstention) | CP@5 | Faithfulness (Direct) | Hallucination (Direct) | Answer Relevance |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cat in ("factoid", "multi_condition", "ambiguous", "unanswerable"):
        grp = [r for r in judged if r["question_category"] == cat]
        direct = [r for r in grp if not r["is_abstention"]]
        n_abs = len(grp) - len(direct)
        hallu_rate = round(_m([float(r["has_hallucination"]) for r in direct]) * 100.0, 2) if direct else 0.0
        lines.append(
            f"| `{cat}` | {len(grp)} | {len(direct)} ({len(direct)*100.0/max(1, len(grp)):.1f}%) | "
            f"{n_abs} ({n_abs*100.0/max(1, len(grp)):.1f}%) | {_m([r['context_precision'] for r in grp]):.4f} | "
            f"{_m([r['faithfulness'] for r in direct]):.4f} | {hallu_rate:.2f}% | "
            f"{_m([r['answer_relevance'] for r in grp]):.4f} |"
        )
    amb_grp = [r for r in judged if r["question_category"] == "ambiguous"]
    amb_ok = sum(int(r.get("handles_ambiguity", 0)) for r in amb_grp)
    lines.extend(
        [
            "",
            f"- **Tỷ lệ nhận diện & làm rõ sự mơ hồ (`ambiguous`):** **{amb_ok}/{len(amb_grp)} ({amb_ok*100.0/max(1, len(amb_grp)):.2f}%)**",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"n_total": len(judged), "report_path": str(report_path)}


def chay_danh_gia_counterfactual(
    cf_path: Path,
    out_gen_path: Path,
    out_judge_path: Path,
    report_path: Path,
    generator_model: str = "qwen3.8:27b",
    judge_model: str = "gemini-3.5-flash",
    gen_delay_sec: float = 0.0,
    judge_delay_sec: float = 0.8,
    batch_size: int = 5,
    llm_call_fn: Optional[Callable[[Sequence[Dict[str, str]], str], Tuple[str, Dict]]] = None,
    gemini_call_fn: Optional[Callable[[str, str], Union[Dict, List]]] = None,
) -> Dict:
    """Chạy thí nghiệm Phản thực tế (30 câu) đo Counterfactual Adherence Rate."""
    rows, _ = doc_danh_sach_jsonl(cf_path)
    out_gen_path.parent.mkdir(parents=True, exist_ok=True)
    call_llm = llm_call_fn or (lambda msgs, m: goi_groq_chat_completion(msgs, model_name=m))
    call_gem = gemini_call_fn or (lambda prompt, m: goi_gemini_json(prompt, model_name=m))

    existing_gen: Dict[str, Dict] = {}
    if out_gen_path.exists():
        for r in doc_danh_sach_jsonl(out_gen_path)[0]:
            if r.get("qid_moi") and r.get("generated_answer"):
                existing_gen[r["qid_moi"]] = r

    gen_rows: List[Dict] = []
    with open(out_gen_path, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, start=1):
            qid = row["qid_moi"]
            if qid in existing_gen:
                rec = dict(existing_gen[qid])
                gen_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            ctx = "\n\n".join(row["counterfactual_contexts"])
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are an e-commerce product question answering assistant. "
                        "Answer the customer's question concisely in 1 to 3 sentences using ONLY the provided product information. "
                        "Do not rely on outside memory if it conflicts with the provided snippet."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Product ASIN: {row['asin']}\n\nRetrieved Product Review Snippets:\n[Snippet 1]\n{ctx}\n\nCustomer Question: {row['question_text_clean']}",
                },
            ]
            ans, usage = call_llm(msgs, generator_model)
            rec = dict(row)
            rec["generated_answer"] = ans
            rec["generator_model"] = generator_model
            rec["usage"] = usage
            gen_rows.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if gen_delay_sec > 0 and llm_call_fn is None and idx < len(rows):
                time.sleep(gen_delay_sec)

    existing_j: Dict[str, Dict] = {}
    if out_judge_path.exists():
        for r in doc_danh_sach_jsonl(out_judge_path)[0]:
            if r.get("qid_moi") and "adheres_counterfactual" in r:
                existing_j[r["qid_moi"]] = r

    to_judge = [r for r in gen_rows if r["qid_moi"] not in existing_j]
    if to_judge:
        with open(out_judge_path, "a", encoding="utf-8") as fa:
            for start in range(0, len(to_judge), batch_size):
                batch = to_judge[start : start + batch_size]
                payload = [
                    {
                        "qid_moi": it["qid_moi"],
                        "question": it["question_text_clean"],
                        "original_2019_fact": it["original_fact"],
                        "counterfactual_context_fact": it["counterfactual_fact"],
                        "generated_answer": it["generated_answer"],
                    }
                    for it in batch
                ]
                prompt = (
                    "Evaluate whether `generated_answer` strictly follows `counterfactual_context_fact` instead of `original_2019_fact`.\n"
                    "Return a JSON array with exact fields:\n"
                    "`qid_moi`, `adheres_counterfactual` (int 1 if answer clearly follows the counterfactual context, else 0), "
                    "`reverts_to_prior` (int 1 if answer repeats the old 2019 fact contradicting context, else 0).\n"
                    f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
                )
                res_list = call_gem(prompt, judge_model)
                if isinstance(res_list, dict):
                    res_list = res_list.get("items") or res_list.get("results") or [res_list]
                bmap = {str(o.get("qid_moi", "")): o for o in (res_list or []) if isinstance(o, dict)}
                for r in batch:
                    j = bmap.get(r["qid_moi"], {})
                    is_fallback = not isinstance(j, dict) or j.get("adheres_counterfactual") is None
                    jrec = {
                        "qid_moi": r["qid_moi"],
                        "adheres_counterfactual": 0 if is_fallback else int(j.get("adheres_counterfactual", 0)),
                        "reverts_to_prior": 0 if is_fallback else int(j.get("reverts_to_prior", 0)),
                        "judge_parse_fallback": is_fallback,
                        "generated_answer": r["generated_answer"],
                    }
                    existing_j[r["qid_moi"]] = jrec
                    fa.write(json.dumps(jrec, ensure_ascii=False) + "\n")
                fa.flush()
                if judge_delay_sec > 0 and gemini_call_fn is None:
                    time.sleep(judge_delay_sec)

    judged = [existing_j[r["qid_moi"]] for r in gen_rows]
    with open(out_judge_path, "w", encoding="utf-8") as fout:
        for rec in judged:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(judged)
    n_adh = sum(int(r["adheres_counterfactual"]) for r in judged)
    n_rev = sum(int(r["reverts_to_prior"]) for r in judged)
    lines = [
        "# Báo cáo Thí nghiệm Phản thực tế (Counterfactual Evaluation — 30 câu AmazonQA dev)",
        "",
        f"- **Generator:** `{generator_model}` | **Judge:** `{judge_model}`",
        f"- **Số câu kiểm thử phản thực tế:** **{n} câu**",
        f"- **Tỷ lệ tuân thủ ngữ cảnh mới (`Counterfactual Adherence Rate`):** **{n_adh}/{n} ({n_adh*100.0/max(1, n):.2f}%)**",
        f"- **Tỷ lệ quay về ký ức cũ 2019 (`Prior Reversion Rate`):** **{n_rev}/{n} ({n_rev*100.0/max(1, n):.2f}%)**",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "n_total": n,
        "adherence_count": n_adh,
        "adherence_rate": round(n_adh * 100.0 / max(1, n), 2),
        "reversion_count": n_rev,
        "report_path": str(report_path),
    }
