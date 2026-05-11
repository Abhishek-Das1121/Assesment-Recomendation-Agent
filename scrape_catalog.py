"""
scrape_catalog.py — One-time script to scrape SHL Individual Test Solutions
into catalog.json. Run this ONCE before starting the server.

Usage:
    python scrape_catalog.py

Output:
    catalog.json  (array of assessment objects)

Each object:
{
    "name": "...",
    "url": "https://www.shl.com/products/product-catalog/view/...",
    "test_type": "K",          # primary type code(s), comma-joined e.g. "K,S"
    "keys": ["Knowledge & Skills", "Simulations"],
    "duration": "9 minutes",   # or "" if untimed/unknown
    "languages": ["English (USA)", "French", ...],
    "description": "..."       # from product page if available
}

The scraper hits the SHL catalog listing page (paginated), 
then fetches each individual product page for full detail.
"""

import json
import time
import re
import sys
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Map display keys → single-letter type codes used by evaluator traces
KEY_TO_CODE = {
    "ability & aptitude": "A",
    "assessment exercises": "E",
    "biodata & situational judgment": "B",
    "competencies": "C",
    "development & 360": "D",
    "knowledge & skills": "K",
    "personality & behavior": "P",
    "simulations": "S",
}


def get_type_codes(keys: list[str]) -> str:
    """Convert list of key display names to comma-joined type codes."""
    codes = []
    for k in keys:
        code = KEY_TO_CODE.get(k.lower().strip())
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


def fetch_catalog_page(session: requests.Session, page: int = 0) -> BeautifulSoup:
    """Fetch one page of the catalog listing (start=page*12)."""
    params = {
        "type": "1",          # Individual Test Solutions filter
        "start": page * 12,
    }
    resp = session.get(CATALOG_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    """Extract product cards from a catalog listing page."""
    products = []
    # SHL catalog uses a table-based or div-based listing; 
    # products appear in <div class="product-catalogue-training-calendar__row">
    # or similar. We look for all links containing /product-catalog/view/
    for a in soup.find_all("a", href=re.compile(r"/products/product-catalog/view/")):
        href = a.get("href", "")
        url = urljoin(BASE_URL, href)
        name = a.get_text(strip=True)
        if name and url not in [p["url"] for p in products]:
            products.append({"name": name, "url": url})
    return products


def fetch_product_detail(session: requests.Session, url: str) -> dict:
    """Fetch and parse a single product detail page."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract duration
        duration = ""
        for tag in soup.find_all(string=re.compile(r"\d+\s+minutes?", re.I)):
            m = re.search(r"\d+\s+minutes?", tag, re.I)
            if m:
                duration = m.group(0)
                break

        # Extract languages — look for "Languages" section
        languages = []
        lang_section = soup.find(string=re.compile(r"language", re.I))
        if lang_section:
            parent = lang_section.find_parent()
            if parent:
                lang_text = parent.get_text(separator=", ", strip=True)
                # crude split
                langs = [l.strip() for l in re.split(r"[,\n]", lang_text) if len(l.strip()) > 2]
                languages = langs[:40]

        # Extract test type keys
        keys = []
        for tag in soup.find_all(string=re.compile(
            r"ability & aptitude|knowledge & skills|personality|simulation|biodata|competencies|development",
            re.I
        )):
            text = tag.strip()
            for key_name in KEY_TO_CODE:
                if key_name in text.lower() and key_name.title() not in keys:
                    keys.append(key_name.title())

        # Description: first substantial paragraph
        description = ""
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 60:
                description = text[:400]
                break

        return {
            "duration": duration,
            "languages": languages,
            "keys": keys,
            "description": description,
        }
    except Exception as e:
        print(f"  Warning: detail fetch failed for {url}: {e}")
        return {"duration": "", "languages": [], "keys": [], "description": ""}


def scrape_all() -> list[dict]:
    session = requests.Session()
    all_products = []
    seen_urls = set()

    print("Scraping catalog listing pages...")
    page = 0
    while True:
        print(f"  Page {page}...")
        soup = fetch_catalog_page(session, page)
        products = parse_listing_page(soup)
        if not products:
            print("  No more products found.")
            break
        new = [p for p in products if p["url"] not in seen_urls]
        if not new:
            break
        for p in new:
            seen_urls.add(p["url"])
        all_products.extend(new)
        page += 1
        time.sleep(0.5)

    print(f"Found {len(all_products)} products. Fetching details...")
    catalog = []
    for i, p in enumerate(all_products):
        print(f"  [{i+1}/{len(all_products)}] {p['name']}")
        detail = fetch_product_detail(session, p["url"])
        entry = {
            "name": p["name"],
            "url": p["url"],
            "test_type": get_type_codes(detail["keys"]),
            "keys": detail["keys"],
            "duration": detail["duration"],
            "languages": detail["languages"],
            "description": detail["description"],
        }
        catalog.append(entry)
        time.sleep(0.3)

    return catalog


if __name__ == "__main__":
    catalog = scrape_all()
    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Wrote {len(catalog)} entries to catalog.json")
