"""Kiểm thử cho module thu thập kho tài liệu chính sách src/policy_corpus.py."""

import json
from pathlib import Path

from src.policy_corpus import (
    gom_doan_chinh_sach,
    trich_xuat_noi_dung_html,
    xay_dung_kho_chinh_sach,
)


def test_trich_xuat_noi_dung_html_bo_nav_footer():
    html = """
    <html>
      <head><title>Shopify Terms</title><style>.x{}</style></head>
      <body>
        <nav><a href="/">Home</a></nav>
        <main>
          <h1>Terms of Service</h1>
          <p>Merchants must be at least 18 years old &amp; comply with all laws.</p>
          <p>Fee is &lt; 3% for standard cards and 5*3 cm label size.</p>
        </main>
        <footer>Copyright Shopify</footer>
      </body>
    </html>
    """
    title, blocks = trich_xuat_noi_dung_html(html)
    assert title == "Shopify Terms"
    joined = " ".join(blocks)
    assert "Home" not in joined
    assert "Copyright Shopify" not in joined
    assert "< 3%" in joined
    assert "5*3 cm" in joined


def test_gom_doan_chinh_sach():
    blocks = ["Section 1", "Short text here.", "Another paragraph with more details about refunds."]
    passages = gom_doan_chinh_sach(blocks, min_passage_words=5, max_passage_words=20)
    assert len(passages) == 1
    assert "Section 1" in passages[0]


def test_xay_dung_kho_chinh_sach_va_loi_http(tmp_path: Path):
    sample_body = " ".join(f"word{i}" for i in range(180))
    html_ok = f"<html><head><title>AUP Policy</title></head><body><main><p>{sample_body}</p></main></body></html>"

    def fake_fetch(url: str):
        if "broken" in url:
            return 404, "Not Found"
        return 200, html_ok

    out_jsonl = tmp_path / "policy.jsonl"
    report = tmp_path / "policy_stats.md"
    res = xay_dung_kho_chinh_sach(
        out_jsonl_path=out_jsonl,
        report_path=report,
        urls=("https://www.shopify.com/legal/aup", "https://www.shopify.com/legal/broken"),
        fetch_fn=fake_fetch,
        min_doc_chars=200,
        delay_sec=0.0,
    )
    assert res["n_docs"] == 1
    assert res["n_failed"] == 1
    assert report.exists()
    rows = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "policy-shopify-aup"
    assert rows[0]["passages"][0]["passage_id"] == "policy-shopify-aup#p000"
