#!/usr/bin/env python3
"""
build_sec_climate_metrics.py

Step 2:
- Read issuer_master.csv (or .csv.gz), which contains:
    cik, gvkey, company_name, tic, cusip, sic, naics
- For each unique CIK, download recent 10-K filings from SEC EDGAR.
- From each filing, extract a simple risk-factor section and compute
  climate-related text metrics.
- Stream results into a slim gzipped CSV:
    clean_data/sec_climate_disclosure_metrics.csv.gz

Designed for:
- Very low RAM: no large DataFrame kept; process company-by-company.
- Very low disk: only small metrics file is saved, no full filings.

You MUST set SEC_USER_AGENT to your info before running
(see README for details).
"""

import csv
import gzip
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ========== CONFIG (CAN BE OVERRIDDEN BY ENV) ==========

# Path to issuer master (step 1 output)
ISSUER_MASTER_PATH = os.environ.get(
    "ISSUER_MASTER_PATH",
    "clean_data/issuer_master.csv",  # or ".csv.gz"
)

# Output metrics (gzipped CSV)
OUTPUT_METRICS_PATH = os.environ.get(
    "OUTPUT_METRICS_PATH",
    "clean_data/sec_climate_disclosure_metrics.csv.gz",
)

# SEC requires a descriptive User-Agent. CHANGE THIS or set SEC_USER_AGENT env.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "CHANGE-ME Your Name Your Affiliation you@example.com",
)

# Max number of 10-K filings per company to process
MAX_10K_PER_COMPANY = int(os.environ.get("MAX_10K_PER_COMPANY", "10"))

# Small sleep between filings to be polite to SEC (per filing)
SLEEP_BETWEEN_FILINGS_SEC = float(os.environ.get("SLEEP_BETWEEN_FILINGS_SEC", "0.2"))

# Optional external keyword file (one term per line)
CLIMATE_KEYWORD_PATH = os.environ.get(
    "CLIMATE_KEYWORD_PATH",
    "data/keywords/climate_keywords_base.txt",
)

# -------- Time-ease controls --------
# Run for TIME_EASE_INTERVAL_SEC, then flush + sleep TIME_EASE_SLEEP_SEC, then continue.
TIME_EASE_INTERVAL_SEC = int(os.environ.get("TIME_EASE_INTERVAL_SEC", str(2 * 60 * 60)))  # default 2 hours
TIME_EASE_SLEEP_SEC = float(os.environ.get("TIME_EASE_SLEEP_SEC", "10"))  # default 10 seconds

# -------- Global rate limit controls --------
# Enforce a minimum gap between ANY SEC HTTP requests to avoid abuse/blocks.
# Default: 1 request/sec (very conservative).
MIN_REQUEST_INTERVAL_SEC = float(os.environ.get("MIN_REQUEST_INTERVAL_SEC", "1.0"))
MAX_429_RETRIES = int(os.environ.get("MAX_429_RETRIES", "5"))

# -------- 403 safety controls --------
# How many 403s on submissions.json we tolerate before deciding we're probably blocked
SUBMISSION_403_LIMIT = int(os.environ.get("SUBMISSION_403_LIMIT", "50"))

WORD_REGEX = re.compile(r"[a-zA-Z]+")


# Baseline keywords (used if external file is missing)
BASELINE_CLIMATE_KEYWORDS: List[str] = [
    "climate",
    "climate change",
    "climatic",
    "global warming",
    "climate risk",
    "transition risk",
    "physical risk",
    "greenhouse",
    "greenhouse gas",
    "ghg",
    "carbon",
    "carbon dioxide",
    "co2",
    "emission",
    "emissions",
    "net zero",
    "decarbonization",
    "decarbonisation",
    "sea level",
    "sea-level rise",
    "flood",
    "flooding",
    "wildfire",
    "heatwave",
    "heat wave",
    "extreme weather",
    "hurricane",
    "typhoon",
    "drought",
    "storm surge",
    "temperature rise",
]

CLIMATE_KEYWORDS: List[str] = BASELINE_CLIMATE_KEYWORDS.copy()

# Global timestamp of last SEC request
LAST_REQUEST_TIME: Optional[float] = None

# Global counter for consecutive 403 on submissions
SUBMISSION_403_COUNT: int = 0


# ========== BASIC HELPERS ==========

def ensure_dir(path: str) -> None:
    """Create parent directory if it does not exist."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def make_sec_session() -> requests.Session:
    """Create a requests session with proper SEC headers."""
    if "CHANGE-ME" in SEC_USER_AGENT:
        raise RuntimeError(
            "Please set SEC_USER_AGENT env var to something like "
            "'Your Name Your Affiliation you@example.com'."
        )

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
    })
    return sess


def sec_get(
    session: requests.Session,
    url: str,
    timeout: float = 30.0,
) -> Optional[requests.Response]:
    """
    Rate-limited GET wrapper with 429 (Too Many Requests) handling.

    - Enforces MIN_REQUEST_INTERVAL_SEC between ANY two calls
    - On 429 uses backoff & retry
    - All other statuses (including 403) are just returned; callers decide.
    """
    global LAST_REQUEST_TIME

    retries_429 = 0

    while True:
        # Enforce minimum spacing across all SEC requests
        now = time.time()
        if LAST_REQUEST_TIME is not None:
            delta = now - LAST_REQUEST_TIME
            if delta < MIN_REQUEST_INTERVAL_SEC:
                sleep_for = MIN_REQUEST_INTERVAL_SEC - delta
                time.sleep(max(sleep_for, 0.0))

        try:
            resp = session.get(url, timeout=timeout)
        except Exception as e:
            print(f"[ERROR] GET {url}: {e}")
            return None

        LAST_REQUEST_TIME = time.time()
        status = resp.status_code

        # Handle 429 with backoff
        if status == 429:
            retries_429 += 1
            if retries_429 > MAX_429_RETRIES:
                print(f"[ERROR] GET {url}: 429 Too Many Requests after {MAX_429_RETRIES} retries.")
                return None

            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = min(300.0, 2.0 ** retries_429)
            else:
                wait = min(300.0, 2.0 ** retries_429)

            print(
                f"[WARN] 429 Too Many Requests for {url}; "
                f"sleeping {wait:.1f}s (retry {retries_429})..."
            )
            time.sleep(wait)
            continue

        # No special handling for 403 or 5xx here; caller decides.
        return resp


def load_climate_keywords(path: str) -> List[str]:
    """
    Load climate keywords from a text file (one per line).
    Lines starting with '#' are comments. If file is missing,
    fall back to BASELINE_CLIMATE_KEYWORDS.
    """
    p = Path(path)
    if not p.is_file():
        print(f"[WARN] Keyword file {p} not found; using built-in baseline list.")
        return BASELINE_CLIMATE_KEYWORDS.copy()

    keywords: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keywords.append(line.lower())

    print(f"[INFO] Loaded {len(keywords)} climate keywords from {p}")
    return keywords


def fetch_company_submissions(
    session: requests.Session,
    cik: str,
) -> Optional[Dict]:
    """
    Download the SEC submissions JSON for a given CIK (10-digit string).
    Returns dict or None on error.
    """
    global SUBMISSION_403_COUNT

    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    resp = sec_get(session, url, timeout=30.0)
    if resp is None:
        print(f"[ERROR] CIK {cik}: submissions.json request failed (no response).")
        return None

    status = resp.status_code

    # Handle 403 specifically here (soft-skip this issuer, but detect global block)
    if status == 403:
        SUBMISSION_403_COUNT += 1
        print(
            f"[WARN] CIK {cik}: submissions.json returned 403 Forbidden "
            f"(consecutive 403 count = {SUBMISSION_403_COUNT}). Skipping this issuer."
        )
        if SUBMISSION_403_COUNT >= SUBMISSION_403_LIMIT:
            raise RuntimeError(
                f"Hit {SUBMISSION_403_COUNT} consecutive 403s from data.sec.gov. "
                "Likely IP/User-Agent block; stopping to avoid hammering SEC."
            )
        return None

    # Reset 403 streak on any non-403 response
    SUBMISSION_403_COUNT = 0

    if status != 200:
        print(f"[WARN] CIK {cik}: submissions.json status {status}")
        return None

    try:
        return resp.json()
    except Exception as e:
        print(f"[ERROR] CIK {cik}: error parsing submissions.json: {e}")
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


def fetch_filing_text(session: requests.Session, url: str) -> Optional[str]:
    """
    Download a filing document (HTML or TXT) and return plain text.
    """
    resp = sec_get(session, url, timeout=60.0)
    if resp is None:
        print(f"[ERROR] Filing URL {url}: request failed.")
        return None

    status = resp.status_code
    if status == 403:
        print(f"[WARN] Filing URL {url} returned 403 Forbidden. Skipping this filing.")
        return None

    if status != 200:
        print(f"[WARN] Filing URL {url} status {status}")
        return None

    text = resp.text
    content_type = resp.headers.get("Content-Type", "").lower()

    try:
        if "html" in content_type or "<html" in text.lower():
            soup = BeautifulSoup(text, "lxml")
            return soup.get_text(separator=" ", strip=True)
        else:
            return text
    except Exception as e:
        print(f"[ERROR] Parsing filing HTML/TXT at {url}: {e}")
        return None


def extract_risk_factor_section(full_text: str) -> str:
    """
    Very simple heuristic:
    - Try to extract 'Item 1A. Risk Factors' section if present;
    - Otherwise return the full text.

    This keeps things cheap & robust; you can refine later.
    """
    if not full_text:
        return ""

    text_lower = full_text.lower()
    start_match = re.search(r"item\s+1a\.", text_lower)
    if not start_match:
        return full_text

    start_idx = start_match.start()

    # Look for the next item heading
    next_match = re.search(r"item\s+1b\.|item\s+2\.", text_lower[start_idx + 10:])
    if next_match:
        end_idx = start_idx + 10 + next_match.start()
        return full_text[start_idx:end_idx]
    else:
        return full_text[start_idx:]


def compute_climate_metrics(text: str) -> Dict[str, float]:
    """
    Compute simple climate-related metrics from a block of text.
    - total_word_count
    - climate_keyword_count
    - climate_keyword_share
    - climate_phrase_count (for multi-word phrases)
    - climate_phrase_share
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


# ========== CORE ITERATOR (COMPANY → FILINGS → ROWS) ==========

def iter_company_climate_rows(
    cik: str,
    static_info: Dict[str, str],
    session: requests.Session,
) -> Iterable[Dict[str, object]]:
    """
    For a single company (identified by CIK), yield climate metric rows
    for up to MAX_10K_PER_COMPANY 10-K filings.

    static_info: dict with gvkey, company_name, tic, cusip, sic, naics, etc.
    """
    submissions = fetch_company_submissions(session, cik)
    if submissions is None:
        # already logged; just skip this issuer
        return

    filings_10k = extract_10k_filings(submissions)
    if not filings_10k:
        print(f"[INFO] CIK {cik}: no 10-K filings found.")
        return

    # Take most recent filings first
    filings_10k = sorted(filings_10k, key=lambda x: x["filing_date"], reverse=True)
    filings_10k = filings_10k[:MAX_10K_PER_COMPANY]

    for f in filings_10k:
        url = build_filing_url(cik, f["accession_number"], f["primary_document"])
        print(f"[INFO] CIK {cik}: processing {f['form']} {f['accession_number']} {f['filing_date']}")
        text = fetch_filing_text(session, url)
        if text is None:
            continue

        section_text = extract_risk_factor_section(text)
        metrics = compute_climate_metrics(section_text)

        row = {
            # IDs
            "cik": cik.zfill(10),
            "gvkey": static_info.get("gvkey"),
            "company_name_master": static_info.get("company_name"),
            "tic": static_info.get("tic"),
            "cusip": static_info.get("cusip"),
            "sic": static_info.get("sic"),
            "naics": static_info.get("naics"),

            # SEC-level info
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


# ========== MAIN PIPELINE ==========

def load_issuer_master(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load issuer_master (small) and build a dict:
        cik -> {gvkey, company_name, tic, cusip, sic, naics}
    """
    print(f"[INFO] Loading issuer master from {path} ...")
    df = pd.read_csv(path)

    # Normalize CIK to 10-digit strings (in case)
    if "cik" not in df.columns:
        raise KeyError("issuer_master must have a 'cik' column.")

    cik_series = df["cik"].astype(str).str.strip().str.replace(r"\D", "", regex=True)
    cik_series = cik_series.apply(lambda x: x.zfill(10)[-10:] if x else x)
    df["cik_norm"] = cik_series

    cols = df.columns.tolist()
    for col in ["gvkey", "company_name", "tic", "cusip", "sic", "naics"]:
        if col not in cols:
            df[col] = pd.NA

    df_unique = df.dropna(subset=["cik_norm"]).drop_duplicates(subset=["cik_norm"])

    issuer_dict: Dict[str, Dict[str, str]] = {}
    for _, row in df_unique.iterrows():
        cik = row["cik_norm"]
        issuer_dict[cik] = {
            "gvkey": row.get("gvkey"),
            "company_name": row.get("company_name"),
            "tic": row.get("tic"),
            "cusip": row.get("cusip"),
            "sic": row.get("sic"),
            "naics": row.get("naics"),
        }

    print(f"[INFO] Loaded {len(issuer_dict)} unique issuers from issuer_master.")
    return issuer_dict


def main():
    global CLIMATE_KEYWORDS

    # 0. Load keyword list
    CLIMATE_KEYWORDS = load_climate_keywords(CLIMATE_KEYWORD_PATH)

    # 1. Load issuer master into a small dict
    issuer_dict = load_issuer_master(ISSUER_MASTER_PATH)
    if not issuer_dict:
        print("[ERROR] No issuers found in issuer_master.")
        return

    # 1.5 Create shared SEC session
    session = make_sec_session()

    # Time-ease timers
    start_time = time.time()
    last_pause_time = start_time

    # 2. Prepare writer
    ensure_dir(OUTPUT_METRICS_PATH)
    fieldnames = [
        # IDs from issuer master
        "cik",
        "gvkey",
        "company_name_master",
        "tic",
        "cusip",
        "sic",
        "naics",

        # SEC filing info
        "sec_company_name",
        "form",
        "accession_number",
        "filing_date",
        "report_date",
        "primary_document",
        "source_url",
        "section_used",

        # metrics
        "total_word_count",
        "climate_keyword_count",
        "climate_keyword_share",
        "climate_phrase_count",
        "climate_phrase_share",
    ]

    # 3. Iterate companies → filings → rows, stream to gzipped CSV
    with gzip.open(OUTPUT_METRICS_PATH, "wt", newline="", encoding="utf-8") as gzfile:
        writer = csv.DictWriter(gzfile, fieldnames=fieldnames)
        writer.writeheader()

        total_firms = len(issuer_dict)
        try:
            for idx, (cik, static_info) in enumerate(issuer_dict.items(), start=1):
                print(f"\n[INFO] === Firm {idx}/{total_firms} | CIK {cik} ===")

                # ---- time-ease check (every ~2 hours by default) ----
                now = time.time()
                if now - last_pause_time >= TIME_EASE_INTERVAL_SEC:
                    elapsed = now - start_time
                    print(
                        f"[INFO] Time-ease: elapsed {elapsed/3600:.2f} hours "
                        f"since start. Flushing and sleeping for {TIME_EASE_SLEEP_SEC} seconds..."
                    )
                    gzfile.flush()
                    time.sleep(TIME_EASE_SLEEP_SEC)
                    last_pause_time = time.time()
                # ------------------------------------------------------

                try:
                    for row in iter_company_climate_rows(cik, static_info, session):
                        writer.writerow(row)
                except Exception as e:
                    print(f"[ERROR] CIK {cik}: unexpected error {e}")
                    continue
        except RuntimeError as e:
            # This is triggered if we hit too many 403s in a row on submissions
            print(f"[ERROR] {e}")
            print("[INFO] Stopping early to avoid hammering SEC; partial metrics file is still saved.")

    print(f"[INFO] Done. Climate metrics saved to {OUTPUT_METRICS_PATH}")


if __name__ == "__main__":
    main()
