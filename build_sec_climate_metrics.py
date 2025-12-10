#!/usr/bin/env python3
"""
build_sec_climate_metrics_resumable.py

Step 2 (resumable version):
- Read issuer_master.csv (or .csv.gz), which contains:
    cik, gvkey, company_name, tic, cusip, sic, naics
- For each unique CIK, download recent 10-K filings from SEC EDGAR.
- Extract a simple risk-factor section and compute climate text metrics.
- Stream results into a slim gzipped CSV:
    clean_data/sec_climate_disclosure_metrics.csv.gz

Features:
- Resumable: keeps a progress JSON with the last fully processed firm index.
- Time-boxed: stops after MAX_RUNTIME_SECONDS (~2h) so you can run via cron/GitHub Actions.
- Gentle rate limit: per-filing sleep, per-company sleep, and a stop after too many 403s.

IMPORTANT:
- Set SEC_USER_AGENT to your own info (name + affiliation + email).
"""

import csv
import gzip
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ========== CONFIG (EDIT THESE) ==========

# Path to issuer master (step 1 output)
ISSUER_MASTER_PATH = "clean_data/issuer_master.csv"  # or ".csv.gz"

# Output metrics (gzipped CSV)
OUTPUT_METRICS_PATH = "clean_data/sec_climate_disclosure_metrics.csv.gz"

# Progress file (JSON)
PROGRESS_PATH = "clean_data/sec_climate_progress.json"

# SEC requires a descriptive User-Agent. CHANGE THIS.
SEC_USER_AGENT = "Sirui Zhao Cornell University sz695@cornell.edu"

# Max number of 10-K filings per company to process
MAX_10K_PER_COMPANY = 10

# Small sleeps to avoid hammering SEC
SLEEP_BETWEEN_FILINGS_SEC = 0.3
SLEEP_BETWEEN_COMPANIES_SEC = 0.5

# Time-boxed run: stop after this many seconds (2h = 7200; use a bit less to be safe)
MAX_RUNTIME_SECONDS = 7100

# Stop entirely if we hit too many consecutive 403s (likely blocked)
MAX_CONSECUTIVE_403 = 20

# Climate-related keywords / phrases (simple baseline; refine later)
CLIMATE_KEYWORDS = [
    "climate",
    "emission",
    "emissions",
    "carbon",
    "co2",
    "greenhouse",
    "ghg",
    "global warming",
    "net zero",
    "decarbonization",
    "decarbonisation",
    "climate change",
]

WORD_REGEX = re.compile(r"[a-zA-Z]+")


# ========== BASIC HELPERS ==========

def ensure_dir(path: str) -> None:
    """Create parent directory if it does not exist."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def make_sec_session() -> requests.Session:
    """Create a requests session with proper SEC headers."""
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    })
    return sess


def fetch_company_submissions(
    session: requests.Session,
    cik: str,
    state: Dict[str, int],
) -> Optional[Dict]:
    """
    Download the SEC submissions JSON for a given CIK (10-digit string).
    Tracks consecutive 403s in `state`.
    """
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 403:
            state["consecutive_403"] += 1
            print(f"[WARN] CIK {cik}: submissions.json 403 (consecutive_403={state['consecutive_403']})")
            return None
        else:
            state["consecutive_403"] = 0

        if resp.status_code != 200:
            print(f"[WARN] CIK {cik}: submissions.json status {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        print(f"[ERROR] CIK {cik}: error fetching submissions.json: {e}")
        return None


def extract_10k_filings(submissions: Dict) -> List[Dict]:
    """
    From a submissions JSON, extract recent 10-K filings.
    Returns list of dicts with fields needed to fetch the filing document.
    """
    filings = submissions.get("filings", {})
    recent = filings.get("recent", {})
    forms = recent.get("form", [])
    accession_nums = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])
    company_name = submissions.get("name", "")

    ten_ks = []
    for form, acc, fdate, rdate, doc in zip(
        forms, accession_nums, filing_dates, report_dates, primary_docs
    ):
        if form not in ("10-K", "10K"):
            continue
        ten_ks.append({
            "form": form,
            "accession_number": acc,
            "filing_date": fdate,
            "report_date": rdate,
            "primary_document": doc,
            "company_name_sec": company_name,
        })

    return ten_ks


def build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    """
    Construct the URL to the primary filing document in EDGAR Archives.
    CIK in path is numeric; accession number path uses digits only (no dashes).
    """
    cik_int = int(cik)
    acc_no_digits = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_no_digits}/{primary_document}"
    )


def fetch_filing_text(
    session: requests.Session,
    url: str,
    state: Dict[str, int],
) -> Optional[str]:
    """
    Download a filing document (HTML or TXT) and return plain text.
    Tracks consecutive 403s in `state`.
    """
    # Switch Host header for www.sec.gov
    session.headers.update({"Host": "www.sec.gov"})
    try:
        resp = session.get(url, timeout=60)
        if resp.status_code == 403:
            state["consecutive_403"] += 1
            print(f"[WARN] Filing URL 403 (consecutive_403={state['consecutive_403']}): {url}")
            return None
        else:
            state["consecutive_403"] = 0

        if resp.status_code != 200:
            print(f"[WARN] Filing URL {url} status {resp.status_code}")
            return None
        text = resp.text
        content_type = resp.headers.get("Content-Type", "").lower()

        if "html" in content_type or "<html" in text.lower():
            soup = BeautifulSoup(text, "lxml")
            return soup.get_text(separator=" ", strip=True)
        else:
            return text
    except Exception as e:
        print(f"[ERROR] Filing URL {url}: error {e}")
        return None
    finally:
        # Restore Host header for JSON API
        session.headers.update({"Host": "data.sec.gov"})


def extract_risk_factor_section(full_text: str) -> str:
    """
    Very simple heuristic:
    - Try to extract 'Item 1A. Risk Factors' section if present;
    - Otherwise return the full text.
    """
    if not full_text:
        return ""

    text_lower = full_text.lower()
    start_match = re.search(r"item\s+1a\.", text_lower)
    if not start_match:
        return full_text

    start_idx = start_match.start()
    next_match = re.search(r"item\s+1b\.|item\s+2\.", text_lower[start_idx + 10:])
    if next_match:
        end_idx = start_idx + 10 + next_match.start()
        return full_text[start_idx:end_idx]
    else:
        return full_text[start_idx:]


def compute_climate_metrics(text: str) -> Dict[str, float]:
    """
    Compute simple climate-related metrics from a block of text.
    """
    if not text:
        return {
            "total_word_count": 0,
            "climate_keyword_count": 0,
            "climate_keyword_share": 0.0,
            "climate_phrase_count": 0,
            "climate_phrase_share": 0.0,
        }

    text_lower = text.lower()
    tokens = WORD_REGEX.findall(text_lower)
    total_words = len(tokens)

    single_words = [kw for kw in CLIMATE_KEYWORDS if " " not in kw]
    multi_phrases = [kw for kw in CLIMATE_KEYWORDS if " " in kw]

    keyword_count = 0
    for kw in single_words:
        keyword_count += tokens.count(kw)

    phrase_count = 0
    for phrase in multi_phrases:
        phrase_count += len(re.findall(re.escape(phrase), text_lower))

    if total_words > 0:
        keyword_share = keyword_count / total_words
        phrase_share = phrase_count / total_words
    else:
        keyword_share = 0.0
        phrase_share = 0.0

    return {
        "total_word_count": total_words,
        "climate_keyword_count": keyword_count,
        "climate_keyword_share": keyword_share,
        "climate_phrase_count": phrase_count,
        "climate_phrase_share": phrase_share,
    }


# ========== PROGRESS HANDLING ==========

def load_progress(path: str) -> int:
    """
    Load progress JSON. Returns last_index (firm index).
    If no file, returns -1 (start from first firm).
    """
    if not os.path.exists(path):
        print("[INFO] No progress file found; starting from first firm.")
        return -1
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        last_index = int(obj.get("last_index", -1))
        print(f"[INFO] Loaded progress: last_index={last_index}")
        return last_index
    except Exception as e:
        print(f"[WARN] Could not load progress file ({e}); starting from first firm.")
        return -1


def save_progress(path: str, last_index: int, total_firms: int) -> None:
    """
    Save progress JSON with last fully processed firm index.
    """
    ensure_dir(path)
    obj = {
        "last_index": int(last_index),
        "total_firms": int(total_firms),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[INFO] Progress saved: last_index={last_index}/{total_firms}")


# ========== ISSUER MASTER LOADING ==========

def load_issuer_master(path: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Load issuer_master (small) and return a list of (cik, static_info) tuples,
    sorted by cik to get a stable order.

    static_info: {gvkey, company_name, tic, cusip, sic, naics}
    """
    print(f"[INFO] Loading issuer master from {path} ...")
    df = pd.read_csv(path)

    if "cik" not in df.columns:
        raise KeyError("issuer_master must have a 'cik' column.")

    cik_series = df["cik"].astype(str).str.strip().str.replace(r"\D", "", regex=True)
    cik_series = cik_series.apply(lambda x: x.zfill(10)[-10:] if x else x)
    df["cik_norm"] = cik_series

    # ensure expected cols exist
    for col in ["gvkey", "company_name", "tic", "cusip", "sic", "naics"]:
        if col not in df.columns:
            df[col] = pd.NA

    df_unique = df.dropna(subset=["cik_norm"]).drop_duplicates(subset=["cik_norm"])
    rows = []
    for _, row in df_unique.iterrows():
        cik = row["cik_norm"]
        static_info = {
            "gvkey": row.get("gvkey"),
            "company_name": row.get("company_name"),
            "tic": row.get("tic"),
            "cusip": row.get("cusip"),
            "sic": row.get("sic"),
            "naics": row.get("naics"),
        }
        rows.append((cik, static_info))

    # stable order by cik
    rows.sort(key=lambda x: x[0])
    print(f"[INFO] Loaded {len(rows)} unique issuers from issuer_master.")
    return rows


# ========== CORE ITERATOR (COMPANY → FILINGS → ROWS) ==========

def iter_company_climate_rows(
    cik: str,
    static_info: Dict[str, str],
    state: Dict[str, int],
) -> Iterable[Dict[str, object]]:
    """
    For a single company (identified by CIK), yield climate metric rows
    for up to MAX_10K_PER_COMPANY 10-K filings.
    """
    session = make_sec_session()
    submissions = fetch_company_submissions(session, cik, state)
    if submissions is None:
        return

    filings_10k = extract_10k_filings(submissions)
    if not filings_10k:
        print(f"[INFO] CIK {cik}: no 10-K filings found.")
        return

    filings_10k = sorted(filings_10k, key=lambda x: x["filing_date"], reverse=True)
    filings_10k = filings_10k[:MAX_10K_PER_COMPANY]

    for f in filings_10k:
        if state["consecutive_403"] >= MAX_CONSECUTIVE_403:
            print("[ERROR] Too many consecutive 403s; stopping requests to avoid being blocked.")
            return

        url = build_filing_url(cik, f["accession_number"], f["primary_document"])
        print(f"[INFO] CIK {cik}: processing {f['form']} {f['accession_number']} {f['filing_date']}")
        text = fetch_filing_text(session, url, state)
        if text is None:
            continue

        section_text = extract_risk_factor_section(text)
        metrics = compute_climate_metrics(section_text)

        row = {
            # IDs from issuer master
            "cik": cik.zfill(10),
            "gvkey": static_info.get("gvkey"),
            "company_name_master": static_info.get("company_name"),
            "tic": static_info.get("tic"),
            "cusip": static_info.get("cusip"),
            "sic": static_info.get("sic"),
            "naics": static_info.get("naics"),

            # SEC info
            "sec_company_name": f.get("company_name_sec", ""),
            "form": f["form"],
            "accession_number": f["accession_number"],
            "filing_date": f["filing_date"],
            "report_date": f.get("report_date", ""),
            "primary_document": f["primary_document"],
            "source_url": url,
            "section_used": "item1a_or_full",
        }
        row.update(metrics)
        yield row

        time.sleep(SLEEP_BETWEEN_FILINGS_SEC)


# ========== MAIN PIPELINE (TIME-BOXED & RESUMABLE) ==========

def main():
    start_time = time.monotonic()

    # 1. Load issuer list (small)
    issuer_list = load_issuer_master(ISSUER_MASTER_PATH)
    total_firms = len(issuer_list)
    if total_firms == 0:
        print("[ERROR] No issuers in issuer_master.")
        return

    # 2. Load progress
    last_index = load_progress(PROGRESS_PATH)  # index of last COMPLETED firm
    start_index = last_index + 1
    if start_index >= total_firms:
        print("[INFO] All firms already processed according to progress file.")
        return

    # 3. Prepare gzipped output (append if exists, else write header)
    ensure_dir(OUTPUT_METRICS_PATH)
    file_exists = os.path.exists(OUTPUT_METRICS_PATH)

    fieldnames = [
        "cik",
        "gvkey",
        "company_name_master",
        "tic",
        "cusip",
        "sic",
        "naics",
        "sec_company_name",
        "form",
        "accession_number",
        "filing_date",
        "report_date",
        "primary_document",
        "source_url",
        "section_used",
        "total_word_count",
        "climate_keyword_count",
        "climate_keyword_share",
        "climate_phrase_count",
        "climate_phrase_share",
    ]

    mode = "at" if file_exists else "wt"
    with gzip.open(OUTPUT_METRICS_PATH, mode, newline="", encoding="utf-8") as gzfile:
        writer = csv.DictWriter(gzfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        state = {"consecutive_403": 0}
        last_completed = last_index

        # 4. Iterate starting from start_index
        for idx in range(start_index, total_firms):
            cik, static_info = issuer_list[idx]

            elapsed = time.monotonic() - start_time
            if elapsed > MAX_RUNTIME_SECONDS:
                print(f"[INFO] Reached time limit ({elapsed:.1f}s); stopping this run.")
                break

            if state["consecutive_403"] >= MAX_CONSECUTIVE_403:
                print("[ERROR] Too many consecutive 403s; stopping run to avoid being blocked.")
                break

            print(f"\n[INFO] === Firm {idx}/{total_firms - 1} | CIK {cik} ===")

            try:
                any_row = False
                for row in iter_company_climate_rows(cik, static_info, state):
                    writer.writerow(row)
                    any_row = True

                # mark firm as completed regardless of whether we got filings
                last_completed = idx
                save_progress(PROGRESS_PATH, last_completed, total_firms)

            except Exception as e:
                print(f"[ERROR] CIK {cik}: unexpected error {e}")
                # still mark this firm as "done" to avoid infinite retry loops
                last_completed = idx
                save_progress(PROGRESS_PATH, last_completed, total_firms)

            time.sleep(SLEEP_BETWEEN_COMPANIES_SEC)

    print("[INFO] Run finished. You can rerun this script; it will resume from the next firm.")


if __name__ == "__main__":
    main()
