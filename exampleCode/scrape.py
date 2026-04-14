from __future__ import annotations

import json
from pathlib import Path

import requests
import yaml

DATA_URL = "https://ysws.hackclub.com/data.yml"
OUTPUT_PATH = Path(__file__).with_name("data.json")
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}


def scrape_data(url: str = DATA_URL, output_path: Path = OUTPUT_PATH) -> Path:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()

    parsed_data = yaml.safe_load(response.text)
    payload = {
        "source_url": url,
        "data": parsed_data,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = scrape_data()
    print(f"Saved {path}")
