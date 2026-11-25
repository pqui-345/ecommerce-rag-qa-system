"""Module thu thập và chuẩn hóa kho tài liệu chính sách thương mại điện tử công khai (Shopify Legal)."""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.clean import clean_text

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SHOPIFY_POLICY_URLS: Tuple[str, ...] = (
    "https://www.shopify.com/legal/terms",
    "https://www.shopify.com/legal/terms-faq",
    "https://www.shopify.com/legal/aup",
    "https://www.shopify.com/legal/privacy",
    "https://www.shopify.com/legal/privacy/consumers",
    "https://www.shopify.com/legal/privacy/merchants",
    "https://www.shopify.com/legal/privacy/partners",
    "https://www.shopify.com/legal/privacy/visitors",
    "https://www.shopify.com/legal/cookies",
    "https://www.shopify.com/legal/dpa",
    "https://www.shopify.com/legal/dsa",
    "https://www.shopify.com/legal/eu-terms",
    "https://www.shopify.com/legal/p2b",
    "https://www.shopify.com/legal/p2b/legal-notices",
    "https://www.shopify.com/legal/terms-shopify-payments",
    "https://www.shopify.com/legal/terms-shopify-protect",
    "https://www.shopify.com/legal/shop-pay-merchant-terms",
    "https://www.shopify.com/legal/terms-gift-cards",
    "https://www.shopify.com/legal/terms-payfac",
    "https://www.shopify.com/legal/terms-cryptocurrency-credit-reward-program",
    "https://www.shopify.com/legal/api-terms",
    "https://www.shopify.com/legal/app-events-terms",
    "https://www.shopify.com/legal/binding-corporate-rules",
    "https://www.shopify.com/legal/gvtaccesspolicy",
    "https://www.shopify.com/legal/trademarks",
    "https://www.shopify.com/legal/patents",
    "https://www.shopify.com/legal/processor-list",
    "https://www.shopify.com/legal/compliance/reports",
)


class _MainHTMLParser(HTMLParser):
    """Trích xuất tiêu đề và danh sách đoạn văn bên trong thẻ <main> (hoặc <article>)."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg"}
    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "li", "tr", "div", "section"}

    def __init__(self) -> None:
        super().__init__()
        self.in_main = False
        self.saw_main = False
        self.skip_depth = 0
        self.in_title = False
        self.title_parts: List[str] = []
        self.buf: List[str] = []
        self.blocks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t == "title":
            self.in_title = True
        if t == "main":
            self.in_main = True
            self.saw_main = True
        if t in self.SKIP_TAGS:
            self.skip_depth += 1
        elif self.in_main and self.skip_depth == 0 and t in self.BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self.in_title = False
        if t == "main":
            self._flush()
            self.in_main = False
        if t in self.SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
        elif self.in_main and self.skip_depth == 0 and t in self.BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self.in_main and self.skip_depth == 0:
            self.buf.append(data)

    def _flush(self) -> None:
        txt = clean_text("".join(self.buf))
        if txt:
            self.blocks.append(txt)
        self.buf = []


def trich_xuat_noi_dung_html(html_text: str) -> Tuple[str, List[str]]:
    """Tách tiêu đề và các đoạn văn sạch từ HTML trang chính sách."""
    parser = _MainHTMLParser()
    parser.feed(html_text or "")
    parser._flush()
    title = clean_text("".join(parser.title_parts))
    blocks = [b for b in parser.blocks if b]
    return title, blocks


def gom_doan_chinh_sach(
    blocks: Sequence[str],
    min_passage_words: int = 25,
    max_passage_words: int = 180,
) -> List[str]:
    """Gom các tiêu đề ngắn với đoạn nội dung liền sau thành các đoạn chuẩn (passages) có nghĩa."""
    passages: List[str] = []
    cur: List[str] = []
    cur_words = 0

    for blk in blocks:
        w = len(blk.split())
        if not cur:
            cur = [blk]
            cur_words = w
            continue
        if cur_words < min_passage_words or (cur_words + w <= max_passage_words):
            cur.append(blk)
            cur_words += w
        else:
            passages.append("\n\n".join(cur))
            cur = [blk]
            cur_words = w

    if cur:
        passages.append("\n\n".join(cur))
    return passages


def tai_trang_html(url: str, timeout: float = 20.0) -> Tuple[int, str]:
    import httpx

    resp = httpx.get(url, follow_redirects=True, timeout=timeout)
    return resp.status_code, resp.text


def _url_to_slug(url: str) -> str:
    part = url.rstrip("/").split("/legal/")[-1]
    return part.replace("/", "_")


def xay_dung_kho_chinh_sach(
    out_jsonl_path: Path,
    report_path: Path,
    urls: Sequence[str] = SHOPIFY_POLICY_URLS,
    fetch_fn: Optional[Callable[[str], Tuple[int, str]]] = None,
    min_doc_chars: int = 800,
    delay_sec: float = 0.2,
) -> Dict:
    """Tải, trích xuất, chuẩn hóa và đóng gói kho tài liệu chính sách TMĐT kèm passage_id chuẩn."""
    fetcher = fetch_fn or tai_trang_html
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    docs: List[Dict] = []
    failed: List[Dict] = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for idx, url in enumerate(urls, start=1):
            try:
                status, html_text = fetcher(url)
            except Exception as exc:
                failed.append({"url": url, "reason": str(exc)})
                continue
            if status != 200:
                failed.append({"url": url, "reason": f"HTTP {status}"})
                continue

            title, blocks = trich_xuat_noi_dung_html(html_text)
            passages_text = gom_doan_chinh_sach(blocks)
            full_text = "\n\n".join(passages_text)
            if len(full_text) < min_doc_chars:
                failed.append({"url": url, "reason": f"too_short ({len(full_text)} chars)"})
                continue

            slug = _url_to_slug(url)
            doc_id = f"policy-shopify-{slug}"
            passages = [
                {
                    "passage_id": f"{doc_id}#p{p_idx:03d}",
                    "passage_index": p_idx,
                    "n_words": len(pt.split()),
                    "n_chars": len(pt),
                    "text": pt,
                }
                for p_idx, pt in enumerate(passages_text)
            ]

            rec = {
                "doc_id": doc_id,
                "source": "shopify_legal",
                "slug": slug,
                "url": url,
                "title": title,
                "fetched_at": fetched_at,
                "n_passages": len(passages),
                "n_words_clean": len(full_text.split()),
                "n_chars_clean": len(full_text),
                "text_clean": full_text,
                "passages": passages,
            }
            docs.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if delay_sec > 0 and fetch_fn is None and idx < len(urls):
                time.sleep(delay_sec)

    total_passages = sum(d["n_passages"] for d in docs)
    total_words = sum(d["n_words_clean"] for d in docs)
    total_chars = sum(d["n_chars_clean"] for d in docs)
    est_pages = round(total_words / 450.0, 1)
    med_chars = round(float(statistics.median([d["n_chars_clean"] for d in docs])), 1) if docs else 0.0

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Báo cáo Kho Tài liệu Chính sách Thương mại Điện tử (Shopify Legal Corpus)",
        "",
        "## 1. Tổng quan nguồn dữ liệu thứ hai",
        "- Nguồn công khai chính thức: **Shopify Legal & Policy Pages** (`https://www.shopify.com/legal/...`)",
        f"- Thời điểm thu thập: `{fetched_at}`",
        f"- Số tài liệu thu thập thành công: **{len(docs)} / {len(urls)} trang** (bỏ qua/lỗi: {len(failed)})",
        f"- Tổng số đoạn chuẩn có định danh `passage_id`: **{total_passages} passages**",
        f"- Tổng quy mô văn bản sạch: **{total_words:,} từ** / **{total_chars:,} ký tự** (~**{est_pages} trang A4**, trung vị **{med_chars:,} ký tự/tài liệu**)",
        "",
        "## 2. Danh sách tài liệu chính sách đã chuẩn hóa",
        "",
        "| `doc_id` | Tiêu đề | Số passages | Số từ | Số ký tự |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for d in docs:
        lines.append(
            f"| `{d['doc_id']}` | {d['title']} | {d['n_passages']} | {d['n_words_clean']:,} | {d['n_chars_clean']:,} |"
        )
    if failed:
        lines.extend(["", "## 3. URL bị loại hoặc lỗi", ""])
        for f_item in failed:
            lines.append(f"- `{f_item['url']}`: {f_item['reason']}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_docs": len(docs),
        "n_failed": len(failed),
        "total_passages": total_passages,
        "total_words": total_words,
        "est_pages": est_pages,
        "out_jsonl_path": str(out_jsonl_path),
        "report_path": str(report_path),
    }
