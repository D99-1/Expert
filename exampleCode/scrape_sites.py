from __future__ import annotations

import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


DEFAULT_INPUT_PATH = Path(__file__).with_name("data.json")
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}


def _extract_page_text(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    container = soup.find("main") or soup.find("article") or soup.body or soup
    headings = [
        heading.get_text(" ", strip=True)
        for heading in container.find_all(["h1", "h2", "h3", "h4"])[:20]
    ]
    paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in container.find_all("p")]

    text_parts = [title, *headings, *paragraphs]
    content = "\n".join(part for part in text_parts if part).strip()

    return {
        "page_title": title,
        "site_content": content,
    }


def _fetch_site_content(url: str, timeout: int = 30) -> dict[str, str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    extracted = _extract_page_text(response.text)
    extracted["final_url"] = response.url
    return extracted


def _enrich_records(payload: dict) -> dict:
    for category_name, records in payload.get("data", {}).items():
        for record in records:
            website = record.get("website")
            if not website:
                continue
            
            site_data = _fetch_site_content(website)
            record["website_content"] = site_data["site_content"]
            record["website_page_title"] = site_data["page_title"]
            record["website_final_url"] = site_data["final_url"]

    return payload


def scrape_sites(input_path: Path = DEFAULT_INPUT_PATH) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    enriched = _enrich_records(payload)
    input_path.write_text(json.dumps(enriched, indent=2, default=str), encoding="utf-8")
    return input_path


if __name__ == "__main__":
    path = scrape_sites()
    print(f"Updated {path}")