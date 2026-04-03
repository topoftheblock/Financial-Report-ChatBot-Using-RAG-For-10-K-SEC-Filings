# -*- coding: utf-8 -*-
"""
SEC 10-K Downloader

Downloads:
1. The main 10-K filing as .html

Requires:
Fill in USER_AGENT with your company and your email. E.g., "MyCompany JD@outlook.com"
"""

# %% Libraries

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

import os


# %% Macros
BASE_DIR = "C:/Users/patri/Desktop/Financial-Report-ChatBot-Using-RAG/"

# %%

# --- CONFIGURATION ---
BASE_SEC = "https://www.sec.gov"
BASE_DATA = "https://data.sec.gov"
USER_AGENT = "YourCompany YourEmail@gmail.com"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
DATA_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}
REQUEST_SLEEP_SECONDS = 0.2


@dataclass
class FilingRecord:
    ticker: str
    cik: str
    company_name: str
    form: str
    filing_date: str
    accession_number: str
    primary_document: str
    primary_doc_description: Optional[str]
    sec_filing_url: str
    sec_index_url: str
    local_html_path: Optional[str] = None


# --- HELPER FUNCTIONS ---

def get_json(url: str, headers: Dict[str, str]) -> dict:
    time.sleep(REQUEST_SLEEP_SECONDS)
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_file(url: str, destination: Path, headers: Dict[str, str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(REQUEST_SLEEP_SECONDS)
    with requests.get(url, headers=headers, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with destination.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)


def load_ticker_map() -> Dict[str, dict]:
    url = f"{BASE_SEC}/files/company_tickers.json"
    data = get_json(url, headers=HEADERS)
    ticker_map: Dict[str, dict] = {}
    for _, item in data.items():
        ticker = item["ticker"].upper()
        ticker_map[ticker] = {
            "cik": str(item["cik_str"]).zfill(10),
            "title": item["title"],
        }
    return ticker_map


def normalize_recent_filings_block(block: dict) -> List[dict]:
    cols = [
        "accessionNumber",
        "form",
        "filingDate",
        "primaryDocument",
        "primaryDocDescription",
    ]
    n = len(block.get("accessionNumber", []))
    return [{c: block.get(c, [None] * n)[i] for c in cols} for i in range(n)]


def collect_all_filings_for_cik(cik_10: str) -> dict:
    url = f"{BASE_DATA}/submissions/CIK{cik_10}.json"
    main = get_json(url, headers=DATA_HEADERS)

    rows = normalize_recent_filings_block(main.get("filings", {}).get("recent", {}))

    for file_info in main.get("filings", {}).get("files", []) or []:
        extra_url = f"{BASE_DATA}/submissions/{file_info['name']}"
        extra = get_json(extra_url, headers=DATA_HEADERS)
        rows.extend(normalize_recent_filings_block(extra))

    return {"name": main.get("name", ""), "filings": rows}


def build_html_url(cik_10: str, accession_number: str, primary_document: str) -> str:
    acc_nodash = accession_number.replace("-", "")
    cik_no_zeros = str(int(cik_10))
    return f"{BASE_SEC}/Archives/edgar/data/{cik_no_zeros}/{acc_nodash}/{primary_document}"


def build_index_url(cik_10: str, accession_number: str) -> str:
    acc_nodash = accession_number.replace("-", "")
    cik_no_zeros = str(int(cik_10))
    return f"{BASE_SEC}/Archives/edgar/data/{cik_no_zeros}/{acc_nodash}/"


# --- MAIN WRAPPER FUNCTION ---

def download_10ks(
    tickers: List[str],
    start_year: int = 2020,
    end_year: int = 2025,
    output_dir: str = "data"
):
    """
    Call this function to run the downloader.

    Example:
        download_10ks(['AAPL', 'MSFT'], 2022, 2023)
    """
    out_path = Path(output_dir)
    ticker_map = load_ticker_map()
    all_records: List[FilingRecord] = []

    for ticker in [t.upper() for t in tickers]:
        info = ticker_map.get(ticker)
        if not info:
            print(f"[WARN] Ticker {ticker} not found.")
            continue

        print(f"--- Processing {ticker} ---")

        try:
            data = collect_all_filings_for_cik(info["cik"])
            seen_accessions = set()

            for row in data["filings"]:
                f_date = row.get("filingDate")
                form = row.get("form")
                accession = row.get("accessionNumber")
                primary_document = row.get("primaryDocument")

                if not f_date or not accession or not primary_document:
                    continue

                if form != "10-K":
                    continue

                filing_year = int(f_date[:4])
                if not (start_year <= filing_year <= end_year):
                    continue

                if accession in seen_accessions:
                    continue
                seen_accessions.add(accession)

                html_url = build_html_url(info["cik"], accession, primary_document)
                index_url = build_index_url(info["cik"], accession)

                html_folder = out_path / "raw" / ticker / str(filing_year)

                local_html_file = html_folder / f"{f_date}_{accession}_10-K.html"

                # Download HTML
                print(f"  [DOWN] {f_date} HTML")
                try:
                    download_file(html_url, local_html_file, HEADERS)
                except Exception as e:
                    print(f"    [ERROR] HTML download failed: {e}")
                    local_html_file = None

                all_records.append(
                    FilingRecord(
                        ticker=ticker,
                        cik=info["cik"],
                        company_name=data["name"],
                        form="10-K",
                        filing_date=f_date,
                        accession_number=accession,
                        primary_document=primary_document,
                        primary_doc_description=row.get("primaryDocDescription"),
                        sec_filing_url=html_url,
                        sec_index_url=index_url,
                        local_html_path=str(local_html_file) if local_html_file else None,
                    )
                )

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    csv_file = out_path / "metadata.csv"
    if all_records:
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_records[0].__dict__.keys())
            writer.writeheader()
            for r in all_records:
                writer.writerow(r.__dict__)

        print(f"\nDone! Files saved to {out_path}. Metadata at {csv_file}")
    else:
        print("\nDone, but no 10-K filings were found for the requested tickers/year range.")


# --- HOW TO RUN IN SPYDER ---
if __name__ == "__main__":
    my_tickers = ["BA", "NVDA", "AAPL", "WMT"]
    download_10ks(tickers=my_tickers, start_year=2025, end_year=2025, output_dir = os.path.join(BASE_DIR, "data/"))