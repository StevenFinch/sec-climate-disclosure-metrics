#!/usr/bin/env python3
"""
build_sec_climate_metrics.py

Step 2: SEC 10-K climate disclosure metrics (resumable, chunked, with sentiment).

- Input: clean_data/issuer_master.csv (or .csv.gz)
    columns: cik, gvkey, company_name, tic, cusip, sic, naics

- For each unique CIK:
    * download recent 10-K filings from SEC EDGAR
    * extract a Risk Factors section (Item 1A) or full text
    * compute:
        - climate keyword / phrase metrics
        - sentence-level climate sentiment (positive/negative/neutral)

- Output per run:
    clean_data/sec_climate_disclosure_metrics_XXXXX_YYYYY.csv.gz

    where XXXXX = first firm index (1-based) processed in this run,
          YYYYY = last firm index (1-based) processed in this run.

- Resume logic:
    * Progress is tracked in clean_data/sec_climate_progress.json
      with fields: last_index (0-based), total_firms, updated_at.
    * Each run starts from last_index + 1 and moves forward.
    * If no more firms, script exits quickly.

IMPORTANT:
- You must set SEC_CONTACT_EMAIL (or SEC_USER_AGENT) in env (we use
  a generic UA with your email from GitHub Secrets).
- Designed to be called repeatedly (e.g. via GitHub Actions every 2h).
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

# Optional: HuggingFace transformers for sentiment
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


# ========== PATHS ANCHORED IN REPO ==========

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DATA_DIR = BASE_DIR / "clean_data"

ISSUER_MASTER_PATH = CLEAN_DATA_DIR / "issuer_master.csv"  # or csv.gz
PROGRESS_PATH = CLEAN_DATA_DIR / "sec_climate_progress.json"

# Chunked output files: sec_climate_disclosure_metrics_XXXXX_YYYYY.csv.gz
SEC_CLIMATE_PREFIX = "sec_climate_disclosure_metrics"


# ========== CONFIG ==========

# Contact email from env (e.g. GitHub Secret)
SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "research-contact@example.com")
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    f"AcademicResearchBot/1.0 (contact: {SEC_CONTACT_EMAIL})",
)

# Max number of 10-K filings per company
MAX_10K_PER_COMPANY = 10

# Gentle rate limiting
SLEEP_BETWEEN_FILINGS_SEC = 0.3
SLEEP_BETWEEN_COMPANIES_SEC = 0.5

# Time-boxed run (per script invocation)
MAX_RUNTIME_SECONDS = 7100  # ~2 hours

# Stop if too many consecutive 403s from SEC
MAX_CONSECUTIVE_403 = 20

# Climate keywords (simple baseline)
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
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[\.\?\!])\s+")


# ========== BASIC HELPERS ==========

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def make_sec_session() -> requests.Session:
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
        session.headers.update({"Host": "data.sec.gov"})


def extract_risk_factor_section(full_text: str) -> str:
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


# ========== SENTIMENT PIPELINE ==========

def build_sentiment_pipeline():
    """
    Use a generic sentiment model that actually produces pos/neg/neu.
    Default: cardiffnlp/twitter-roberta-base-sentiment-latest
    """
    if not HAVE_TRANSFORMERS:
        print("[WARN] transformers not installed -> sentiment disabled.")
        return None

    model_name = os.getenv(
        "SENTIMENT_MODEL_NAME",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    try:
        print(f"[INFO] Loading sentiment model: {model_name}")
        clf = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
        )
        print("[INFO] Sentiment pipeline loaded.")
        return clf
    except Exception as e:
        print(f"[WARN] Failed to load sentiment model ({e}) -> sentiment disabled.")
        return None


def compute_climate_sentiment(text: str, sentiment_pipe) -> Dict[str, float]:
    if sentiment_pipe is None or not text:
        return {
            "climate_sent_n_sents": 0,
            "climate_sent_pos_count": 0,
            "climate_sent_neg_count": 0,
            "climate_sent_neu_count": 0,
            "climate_sent_pos_share": 0.0,
            "climate_sent_neg_share": 0.0,
            "climate_sent_neu_share": 0.0,
            "climate_sent_score_avg": 0.0,
        }

    sentences = SENTENCE_SPLIT_REGEX.split(text)
    climate_sents = []
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped:
            continue
        s_low = s_stripped.lower()
        if any(kw in s_low for kw in CLIMATE_KEYWORDS):
            climate_sents.append(s_stripped)

    n = len(climate_sents)
    if n == 0:
        return {
            "climate_sent_n_sents": 0,
            "climate_sent_pos_count": 0,
            "climate_sent_neg_count": 0,
            "climate_sent_neu_count": 0,
            "climate_sent_pos_share": 0.0,
            "climate_sent_neg_share": 0.0,
            "climate_sent_neu_share": 0.0,
            "climate_sent_score_avg": 0.0,
        }

    results = sentiment_pipe(
        climate_sents,
        truncation=True,
        max_length=256,
    )

    pos = neg = neu = 0
    score_sum = 0.0

    for r in results:
        label = (r.get("label") or "").lower()
        # typical labels: "positive", "neutral", "negative" or "LABEL_0" etc
        if "pos" in label:
            pos += 1
            score_sum += 1.0
        elif "neg" in label:
            neg += 1
            score_sum -= 1.0
        else:
            neu += 1

    pos_share = pos / n
    neg_share = neg / n
    neu_share = neu / n
    avg_score = score_sum / n

    return {
        "climate_sent_n_sents": n,
        "climate_sent_pos_count": pos,
        "climate_sent_neg_count": neg,
        "climate_sent_neu_count": neu,
        "climate_sent_pos_share": pos_share,
        "climate_sent_neg_share": neg_share,
        "climate_sent_neu_share": neu_share,
        "climate_sent_score_avg": avg_score,
    }


def compute_climate_metrics(text: str, sentiment_pipe=None) -> Dict[str, float]:
    if not text:
        base = {
            "total_word_count": 0,
            "climate_keyword_count": 0,
            "climate_keyword_share": 0.0,
            "climate_phrase_count": 0,
            "climate_phrase_share": 0.0,
        }
        base.update(compute_climate_sentiment("", sentiment_pipe))
        return base

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

    keyword_share = keyword_count / total_words if total_words > 0 else 0.0
    phrase_share = phrase_count / total_words if total_words > 0 else 0.0

    base = {
        "total_word_count": total_words,
        "climate_keyword_count": keyword_count,
        "climate_keyword_share": keyword_share,
        "climate_phrase_count": phrase_count,
        "climate_phrase_share": phrase_share,
    }
    base.update(compute_climate_sentiment(text, sentiment_pipe))
    return base


# ========== PROGRESS HANDLING ==========

def load_progress(path: Path, total_firms: int) -> int:
    """
    Returns last_index (0-based). -1 if starting fresh.
    """
    if not path.exists():
        print("[INFO] No progress file -> starting from first firm.")
        return -1
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        last_index = int(obj.get("last_index", -1))
        print(f"[INFO] Loaded progress: last_index={last_index}")
        if last_index >= total_firms:
            last_index = total_firms - 1
        return last_index
    except Exception as e:
        print(f"[WARN] Could not load progress file ({e}); starting fresh.")
        return -1


def save_progress(path: Path, last_index: int, total_firms: int) -> None:
    ensure_dir(path)
    obj = {
        "last_index": int(last_index),
        "total_firms": int(total_firms),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[INFO] Progress saved: last_index={last_index}/{total_firms - 1}")


# ========== ISSUER MASTER LOADING ==========

def load_issuer_master(path: Path) -> List[Tuple[str, Dict[str, str]]]:
    print(f"[INFO] Loading issuer master from {path} ...")
    if not path.exists():
        raise FileNotFoundError(f"issuer_master not found at: {path}")
    df = pd.read_csv(path)

    if "cik" not in df.columns:
        raise KeyError("issuer_master must have a 'cik' column.")

    cik_series = df["cik"].astype(str).str.strip().str.replace(r"\D", "", regex=True)
    cik_series = cik_series.apply(lambda x: x.zfill(10)[-10:] if x else x)
    df["cik_norm"] = cik_series

    for col in ["gvkey", "company_name", "tic", "cusip", "sic", "naics"]:
        if col not in df.columns:
            df[col] = pd.NA

    df_unique = df.dropna(subset=["cik_norm"]).drop_duplicates(subset=["cik_norm"])

    rows: List[Tuple[str, Dict[str, str]]] = []
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

    rows.sort(key=lambda x: x[0])
    print(f"[INFO] Loaded {len(rows)} unique issuers from issuer_master.")
    return rows


# ========== COMPANY → FILINGS → ROWS ==========

def iter_company_climate_rows(
    cik: str,
    static_info: Dict[str, str],
    state: Dict[str, int],
    sentiment_pipe,
) -> Iterable[Dict[str, object]]:
    session = make_sec_session()
    submissions = fetch_company_submissions(session, cik, state)
    if submissions is None:
        return

    filings_10k = extract_10k_filings(submissions)
    if not filings_10k:
        print(f"[INFO] CIK {cik}: no 10-K filings.")
        return

    filings_10k = sorted(filings_10k, key=lambda x: x["filing_date"], reverse=True)
    filings_10k = filings_10k[:MAX_10K_PER_C_]()
