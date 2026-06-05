import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ── Config ──────────────────────────────────
EXPERIENCE_URL = "https://www.linkedin.com/in/andrewyng/details/experience/"
OUTPUT_FILE    = "linkedin_experience.txt"
OUTPUT_DIR     = "Experience_Data"
REQUEST_DELAY  = 1.5
SCRAPE_TIMEOUT = 15

LINKEDIN_COOKIE = os.getenv("LINKEDIN_COOKIE", "")
HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.google.com/",
    "Connection":      "keep-alive",
}

# ── Helpers ──────────────────────────────────
def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r" · .*", "", text)
    return text.strip()

def is_current(duration_text):
    return bool(re.search(r"\bpresent\b|\bnow\b|\bcurrent\b", duration_text, re.I))

def is_date_range(text):
    """Returns True if the string looks like a date range or duration snippet."""
    # Matches strings containing years, months, or common month abbreviations + 4-digit years
    has_duration_keywords = bool(re.search(r"\byr\b|\byears\b|\bmos\b|\bmonths\b", text, re.I))
    has_year_digits = bool(re.search(r"\b(19|20)\d{2}\b", text))
    is_present_marker = is_current(text)
    return has_duration_keywords or (has_year_digits and len(text) < 35) or is_present_marker

# ── Step 1: Fetch LinkedIn HTML ──────────────
def fetch_page(url):
    print("[1/3] Fetching LinkedIn page...")
    session = requests.Session()
    if LINKEDIN_COOKIE:
        for pair in LINKEDIN_COOKIE.split(";"):
            if "=" in pair:
                name, _, val = pair.strip().partition("=")
                session.cookies.set(name.strip(), val.strip(), domain=".linkedin.com")
        print("      → Using supplied session cookie")
    else:
        print("      → No cookie set  (LINKEDIN_COOKIE env var is empty)")

    resp = session.get(url, timeout=SCRAPE_TIMEOUT, headers=HEADERS)
    resp.raise_for_status()

    if any(kw in resp.url for kw in ("authwall", "/login", "/signup", "/uas/")):
        raise RuntimeError(
            "LinkedIn redirected to a login page.\n"
            "  Fix: set the LINKEDIN_COOKIE env var with your li_at token.\n"
        )

    print(f"      → HTTP {resp.status_code}  |  {len(resp.text):,} chars received")
    return resp.text

# ── Step 2: Parse Experience Entries ─────────
def _from_html_dom(soup):
    entries = []
    
    sections = soup.find_all(["section", "div"], id=re.compile(r"experience", re.I)) or \
               soup.find_all(["section", "div"], class_=re.compile(r"experience", re.I))
               
    if not sections:
        return entries

    for section in sections:
        for li in section.find_all("li"):
            # Gather all non-empty strings inside this list item block
            all_strings = [clean_text(s.get_text()) for s in li.find_all("span") if s.get_text()]
            all_strings = [s for s in all_strings if s]
            
            if not all_strings:
                continue

            title = "Unknown Title"
            company = "Unknown Company"
            duration = "N/A"

            # 1. Primary Targeted CSS Selection Rule
            t_el = li.find(["h3", "h4", "span"], class_=re.compile(r"title|bold", re.I))
            c_el = li.find("span", class_=re.compile(r"company|org|secondary", re.I))
            d_el = li.find("span", class_=re.compile(r"date|range", re.I))

            if t_el: title = clean_text(t_el.get_text())
            if c_el: company = clean_text(c_el.get_text())
            if d_el: duration = clean_text(d_el.get_text())

            # 2. Hybrid Safety Fallback Layer (Checks structural spans explicitly)
            # If our company parsing caught a date string, or failed, look at text items
            if is_date_range(company) or company == "Unknown Company":
                # Find strings that do not look like date signatures
                clean_pool = [s for s in all_strings if not is_date_range(s) and len(s) > 2]
                if len(clean_pool) >= 2:
                    title = clean_pool[0]
                    company = clean_pool[1]
                elif len(clean_pool) == 1:
                    # If only one clean string exists, it's likely a nested role where the main 
                    # company name sits further up in a parent tag loop. Try to inherit it.
                    title = clean_pool[0]
                    parent_div = li.find_parent("div")
                    if parent_div:
                        parent_spans = [clean_text(s.get_text()) for s in parent_div.find_all("span") if s.get_text()]
                        parent_pool = [s for s in parent_spans if not is_date_range(s) and len(s) > 2]
                        if parent_pool:
                            company = parent_pool[0]

            # 3. Double-check text list to find accurate duration string
            for s in all_strings:
                if is_date_range(s) and ("-" in s or is_current(s)):
                    duration = s
                    break

            # If the title accidentally caught a date string, drop it
            if is_date_range(title):
                continue

            if title != "Unknown Title" and company != "Unknown Company" and not is_date_range(company):
                entries.append({
                    "company":  company,
                    "title":    title,
                    "duration": duration,
                    "current":  is_current(duration),
                })
    return entries

def parse_experience(html):
    print("[2/3] Parsing experience data...")
    soup    = BeautifulSoup(html, "html.parser")
    entries = _from_html_dom(soup)
    
    # De-duplicate entries cleanly
    seen, unique = set(), []
    for e in entries:
        comp_str = str(e["company"]).lower().strip()
        title_str = str(e["title"]).lower().strip()

        key = (comp_str, title_str)
        if key not in seen and comp_str != "unknown company" and not is_date_range(e["company"]):
            seen.add(key)
            unique.append(e)

    print(f"      → {len(unique)} unique entries parsed")
    return unique

# ── Step 3: Save & Display Report ────────────
def save_report(entries, output_dir, output_file):
    print(f"[3/3] Saving report → {output_dir}/{output_file}")
    os.makedirs(output_dir, exist_ok=True)

    current = [e for e in entries if     e["current"]]
    past    = [e for e in entries if not e["current"]]

    def current_block(e):
        return (
            f"  Company  : {e['company']}\n"
            f"  Title    : {e['title']}\n"
        )

    def past_block(e):
        return (
            f"  Company  : {e['company']}\n"
            f"  Title    : {e['title']}\n"
            f"  Duration : {e['duration']}\n"
        )

    report = "\n".join([
        "LINKEDIN EXPERIENCE REPORT",
        "=" * 80,
        f"Scraped  : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        "=" * 80,
        "",
        f"── CURRENT POSITION(S)  ({len(current)} found) " + "─" * 32,
        "",
        *([current_block(e) for e in current] or ["  (none detected)\n"]),
        f"── PAST POSITIONS  ({len(past)} found) " + "─" * 37,
        "",
        *([past_block(e) for e in past]    or ["  (none detected)\n"]),
    ])

    filepath = os.path.join(output_dir, output_file)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\n✓  Saved → {filepath}")

# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    html = fetch_page(EXPERIENCE_URL)
    time.sleep(REQUEST_DELAY)

    entries = parse_experience(html)

    if not entries:
        print("\n⚠ No experience data found.")
        print("   Make sure to export your LINKEDIN_COOKIE environment variable.")
        raise SystemExit(1)

    save_report(entries, OUTPUT_DIR, OUTPUT_FILE)