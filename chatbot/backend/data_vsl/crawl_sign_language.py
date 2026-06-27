
import requests
import json
import time
import os
import sys
from bs4 import BeautifulSoup

BASE_URL = "https://tudienngonngukyhieu.com"
LISTING_URL = f"{BASE_URL}/ngon-ngu-ky-hieu-theo-tu"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
OUTPUT_FILE = "sign_language_data.json"
PROGRESS_FILE = "crawl_progress.json"

session = requests.Session()
session.headers.update(HEADERS)

NO_INFO = {"chưa có thông tin", "không có thông tin", "đang cập nhật", ""}


def is_valid_desc(desc):
    d = desc.strip().lower()
    return bool(d) and d not in NO_INFO


def fetch(url, retries=3):
    for a in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if a < retries - 1:
                time.sleep(1.5 ** a)
    return None


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.flex.flex-col.items-center.overflow-hidden.rounded-lg.border.md\\:flex-row")
    entries = []
    for card in cards:
        link = card.select_one("h3 a")
        if not link:
            continue
        href = link.get("href", "")
        if not href:
            continue
        e = {
            "word": link.get_text(strip=True),
            "url": BASE_URL + href,
            "description": "",
            "category": "",
            "region": "",
        }
        d = card.select_one("p.text-gray-500")
        if d:
            e["description"] = d.get_text(strip=True)
        c = card.select_one("a[href^='/phan-loai/']")
        if c:
            e["category"] = c.get_text(strip=True)
        r = card.select_one("a[href^='/vung-mien/']")
        if r:
            e["region"] = r.get_text(strip=True)
        entries.append(e)
    return entries


def crawl_listing(max_pages=778):
    all_data = []
    seen = set()
    start_page = 1

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            prog = json.load(f)
            all_data = prog["data"]
            seen = set(item["url"] for item in all_data)
            start_page = prog["next_page"]
        print(f"[*] Resuming from page {start_page} ({len(all_data)} words done)")
    else:
        print(f"[*] Crawling pages 1 to {max_pages}...")

    for page in range(start_page, max_pages + 1):
        url = f"{LISTING_URL}?page={page}" if page > 1 else LISTING_URL
        html = fetch(url)
        if not html:
            print(f"  [!] Page {page} failed, skipping")
            continue

        entries = parse_page(html)
        if not entries:
            print(f"  [*] Page {page} has no entries, stopping")
            break

        new = 0
        for e in entries:
            if e["url"] not in seen:
                seen.add(e["url"])
                all_data.append(e)
                new += 1

        if page % 50 == 0 or page == max_pages or page == start_page:
            print(f"  [>] Page {page}/{max_pages} - {new} new, total {len(all_data)}", flush=True)

        if page % 50 == 0:
            with open(PROGRESS_FILE, "w") as f:
                json.dump({"next_page": page + 1, "data": all_data}, f, ensure_ascii=False)

        time.sleep(0.3)

    with open(PROGRESS_FILE, "w") as f:
        json.dump({"next_page": max_pages + 1, "data": all_data}, f, ensure_ascii=False)

    print(f"[✓] Phase 1: {len(all_data)} words crawled", flush=True)
    return all_data


def enrich_descriptions(data):
    need = [(i, item) for i, item in enumerate(data) if not is_valid_desc(item["description"])]
    print(f"[*] Phase 2: enriching {len(need)} entries...", flush=True)

    updated = 0
    for idx, (i, item) in enumerate(need):
        html = fetch(item["url"])
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for h3 in soup.find_all("h3"):
                if "Cách làm ký hiệu" in h3.get_text():
                    p = h3.find_next_sibling("p")
                    if p:
                        desc = p.get_text(strip=True)
                        if is_valid_desc(desc):
                            data[i]["description"] = desc
                            updated += 1
                    break
        if (idx + 1) % 500 == 0 or idx == len(need) - 1:
            print(f"  [>] {idx+1}/{len(need)} detail pages - {updated} enriched", flush=True)
        time.sleep(0.15)

    print(f"[✓] Phase 2: {updated} entries enriched", flush=True)
    return data


def save(data):
    filtered = [
        {k: item[k] for k in ["word", "description", "category", "region"]}
        for item in data
        if is_valid_desc(item.get("description", ""))
    ]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] Saved {len(filtered)} words to {OUTPUT_FILE}", flush=True)
    return filtered


def main():
    print("=" * 60, flush=True)
    print("CRAWL TỪ ĐIỂN NGÔN NGỮ KÝ HIỆU", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()

    all_data = crawl_listing(max_pages=778)

    print("[*] Phase 2: skipped (descriptions already on listing pages)")

    # Phase 3: filter & save
    filtered = save(all_data)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"THỐNG KÊ", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Tổng số từ crawl:         {len(all_data)}", flush=True)
    print(f"  Từ có mô tả đầy đủ:       {len(filtered)}", flush=True)
    print(f"  Từ không có mô tả:        {len(all_data) - len(filtered)}", flush=True)
    print(f"  Thời gian:                {elapsed/60:.1f} phút", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
