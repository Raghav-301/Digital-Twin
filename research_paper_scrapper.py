import os
import re
import time
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────
DBLP_XML_URL  = "https://dblp.org/pid/n/AndrewYNg.xml"
PAPERS_FILE   = "research_papers.txt"
TEMP_FILE     = "temporary.txt"
OUTPUT_DIR    = "Data_text"
REQUEST_DELAY = 1.5
SCRAPE_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── Helpers ──────────────────────────────────
def url_to_filename(url):
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^\w\-]", "_", slug)
    return re.sub(r"_+", "_", slug).strip("_")[:120] + ".txt"

# ── Step 1: Fetch URLs from DBLP XML ─────────
def extract_ee_urls(xml_url):
    print(f"[1/3] Fetching DBLP XML...")
    resp = requests.get(xml_url, timeout=SCRAPE_TIMEOUT, headers=HEADERS)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    urls = [ee.text.strip() for ee in root.iter("ee") if (ee.text or "").strip().startswith("http")]
    print(f"      → Found {len(urls)} URLs")
    return urls

# ── Step 2: Scrape title + abstract from a URL ─
ABSTRACT_SELECTORS = [{"property": "twitter:description"}, {"property": "og:description"}, {"name": "description"}]
TITLE_SELECTORS    = [{"property": "og:title"}, {"name": "twitter:title"}]

def scrape_url(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=SCRAPE_TIMEOUT, headers=HEADERS).text, "html.parser")
        abstract = next((t["content"].strip() for a in ABSTRACT_SELECTORS if (t := soup.find("meta", attrs=a)) and t.get("content", "").strip()), None)
        title    = next((t["content"].strip() for a in TITLE_SELECTORS    if (t := soup.find("meta", attrs=a)) and t.get("content", "").strip()), None)
        if not title and (t := soup.find("title")) and t.string:
            title = re.split(r"\s*[|\-]\s*(IEEE|PubMed|Springer|ACM|arXiv|PMLR|Nature)", t.string)[0].strip()
        return title, abstract
    except Exception:
        return None, None

# ── Step 3: Batch scrape ──────────────────────
def scrape_all_papers(urls, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total = len(urls)
    print(f"[3/3] Scraping {total} new papers → {output_dir}/\n")
    ok = skipped = 0
    for idx, url in enumerate(urls, 1):
        title, abstract = scrape_url(url)
        if not abstract:
            print(f"  [{idx:>3}/{total}] ✗  {url[:65]}")
            skipped += 1
        else:
            header = title or url
            with open(os.path.join(output_dir, url_to_filename(url)), "w", encoding="utf-8") as f:
                f.write(f"{header}\n{'=' * 80}\n\n{abstract}")
            print(f"  [{idx:>3}/{total}] ✓  {header[:70]}")
            ok += 1
        time.sleep(REQUEST_DELAY)
    print(f"\nDone! {ok} saved, {skipped} skipped.")

# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    all_urls = extract_ee_urls(DBLP_XML_URL)
    if not all_urls:
        raise SystemExit("No <ee> URLs found.")

    with open(TEMP_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_urls) + "\n")
    print(f"[2/3] Saved {len(all_urls)} URLs → {TEMP_FILE}")

    existing = set(open(PAPERS_FILE).read().splitlines()) if os.path.exists(PAPERS_FILE) else set()
    new_urls = [u for u in all_urls if u not in existing]
    print(f"      → {len(existing)} existing, {len(new_urls)} new\n")

    if new_urls:
        scrape_all_papers(new_urls, OUTPUT_DIR)

    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        f.write(open(TEMP_FILE).read())
    os.remove(TEMP_FILE)
    print(f"Updated {PAPERS_FILE}, deleted {TEMP_FILE}")