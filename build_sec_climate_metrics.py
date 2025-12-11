#!/usr/bin/env python3
"""
build_sec_climate_metrics.py

- Input:  clean_data/issuer_master.csv
          columns: cik, gvkey, company_name, tic, cusip, sic, naics (extra cols ignored)
- Output (per run): clean_data/sec_climate_disclosure_metrics_XXXXX_YYYYY.csv.gz
  where XXXXX = first firm index (1-based), YYYYY = last firm index (1-based)

- Resume: uses clean_data/sec_climate_progress.json
  {
    "last_index": int (0-based index of last COMPLETED firm),
    "total_firms": int,
    "updated_at": "...",
  }

- Each run:
  * Reads issuer_master
  * Reads progress JSON (or starts from -1)
  * Starts at last_index + 1
  * Processes forward until (a) time limit or (b) firms exhausted
  * Writes one new chunk file for that range
  * Updates progress JSON after each firm

- Sentiment:
  * Split risk-factor section into sentences
  * Keep only sentences containing climate keywords
  * Run HF sentiment model on those sentences
  * Count pos/neg/neu + average sentiment score

- Extra:
  * New column climate_sentences: JSON list of sentences that contain
    at least one climate keyword, based on the same filter used for sentiment.
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

# Optional sentiment model
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


# ========== PATHS & CONFIG ==========

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DATA_DIR = BASE_DIR / "clean_data"

ISSUER_MASTER_PATH = CLEAN_DATA_DIR / "issuer_master.csv"
PROGRESS_PATH = CLEAN_DATA_DIR / "sec_climate_progress.json"
SEC_CLIMATE_PREFIX = "sec_climate_disclosure_metrics"

SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "research-contact@example.com")
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    f"AcademicResearchBot/1.0 (contact: {SEC_CONTACT_EMAIL})",
)

MAX_10K_PER_COMPANY = 10
SLEEP_BETWEEN_FILINGS_SEC = 0.3
SLEEP_BETWEEN_COMPANIES_SEC = 0.5
MAX_RUNTIME_SECONDS = 7100           # ~2 hours
MAX_CONSECUTIVE_403 = 20

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


# ========== HELPERS ==========

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
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_digits}/{primary_document}"


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


# ========== SENTIMENT & CLIMATE SENTENCES ==========

def extract_climate_sentences(text: str) -> List[str]:
    """
    Extract all sentences that contain at least one climate keyword.
    Sentences are stripped of leading/trailing whitespace.
    """
    if not text:
        return []

    sentences = SENTENCE_SPLIT_REGEX.split(text)
    climate_sents: List[str] = []
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped:
            continue
        s_low = s_stripped.lower()
        if any(kw in s_low for kw in CLIMATE_KEYWORDS):
            climate_sents.append(s_stripped)
    return climate_sents


def build_sentiment_pipeline():
    if not HAVE_TRANSFORMERS:
        print("[WARN] transformers not installed -> sentiment disabled.")
        return None

    model_name = os.getenv("SENTIMENT_MODEL_NAME", "yiyanghkust/finbert-tone")
    try:
        print(f"[INFO] Loading sentiment model: {model_name}")
        clf = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
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

    # Use the same sentence set everywhere (also used for saving to column)
    climate_sents = extract_climate_sentences(text)

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
        # FinBERT-tone labels: "positive", "negative", "neutral"
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


# ========== PROGRESS & ISSUER MASTER ==========

def load_progress(path: Path, total_firms: int) -> int:
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


# ========== COMPANY → ROWS ==========

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
    filings_10k = filings_10k[:MAX_10K_PER_COMPANY]

    for f in filings_10k:
        if state["consecutive_403"] >= MAX_CONSECUTIVE_403:
            print("[ERROR] Too many consecutive 403s; stopping to avoid block.")
            return

        url = build_filing_url(cik, f["accession_number"], f["primary_document"])
        print(f"[INFO] CIK {cik}: {f['form']} {f['accession_number']} {f['filing_date']}")
        text = fetch_filing_text(session, url, state)
        if text is None:
            continue

        section_text = extract_risk_factor_section(text)
        metrics = compute_climate_metrics(section_text, sentiment_pipe)

        # NEW: capture the exact sentences that contain climate keywords
        climate_sents = extract_climate_sentences(section_text)
        climate_sents_str = json.dumps(climate_sents, ensure_ascii=False)

        row = {
            "cik": cik.zfill(10),
            "gvkey": static_info.get("gvkey"),
            "company_name_master": static_info.get("company_name"),
            "tic": static_info.get("tic"),
            "cusip": static_info.get("cusip"),
            "sic": static_info.get("sic"),
            "naics": static_info.get("naics"),
            "sec_company_name": f.get("company_name_sec", ""),
            "form": f["form"],
            "accession_number": f["accession_number"],
            "filing_date": f["filing_date"],
            "report_date": f.get("report_date", ""),
            "primary_document": f["primary_document"],
            "source_url": url,
            "section_used": "item1a_or_full",
            "climate_sentences": climate_sents_str,
        }
        row.update(metrics)
        yield row

        time.sleep(SLEEP_BETWEEN_FILINGS_SEC)


# ========== MAIN (ONE CHUNK) ==========

def main():
    start_time = time.monotonic()

    issuer_list = load_issuer_master(ISSUER_MASTER_PATH)
    total_firms = len(issuer_list)
    if total_firms == 0:
        print("[ERROR] No issuers found in issuer_master.")
        return

    last_index = load_progress(PROGRESS_PATH, total_firms)
    start_index = last_index + 1

    if start_index >= total_firms:
        print("[INFO] All firms already processed according to progress file.")
        return

    sentiment_pipe = build_sentiment_pipeline()

    chunk_start_idx = start_index
    temp_path = CLEAN_DATA_DIR / f"{SEC_CLIMATE_PREFIX}_current.csv.gz"
    ensure_dir(temp_path)

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
        "climate_sent_n_sents",
        "climate_sent_pos_count",
        "climate_sent_neg_count",
        "climate_sent_neu_count",
        "climate_sent_pos_share",
        "climate_sent_neg_share",
        "climate_sent_neu_share",
        "climate_sent_score_avg",
        "climate_sentences",   # NEW COLUMN
    ]

    state = {"consecutive_403": 0}
    last_completed = last_index
    processed_any_firm = False

    with gzip.open(temp_path, "wt", newline="", encoding="utf-8") as gzfile:
        writer = csv.DictWriter(gzfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(start_index, total_firms):
            elapsed = time.monotonic() - start_time
            if elapsed > MAX_RUNTIME_SECONDS:
                print(f"[INFO] Time limit reached ({elapsed:.1f}s); stopping this run.")
                break
            if state["consecutive_403"] >= MAX_CONSECUTIVE_403:
                print("[ERROR] Too many consecutive 403s; stopping run.")
                break

            cik, static_info = issuer_list[idx]
            processed_any_firm = True
            print(f"\n[INFO] === Firm {idx}/{total_firms - 1} | CIK {cik} ===")

            try:
                for row in iter_company_climate_rows(cik, static_info, state, sentiment_pipe):
                    writer.writerow(row)
                last_completed = idx
                save_progress(PROGRESS_PATH, last_completed, total_firms)
            except Exception as e:
                print(f"[ERROR] CIK {cik}: unexpected error {e}")
                last_completed = idx
                save_progress(PROGRESS_PATH, last_completed, total_firms)

            time.sleep(SLEEP_BETWEEN_COMPANIES_SEC)

    if not processed_any_firm or last_completed < chunk_start_idx:
        print("[INFO] No new firms processed this run; removing temp chunk.")
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        return

    start_human = chunk_start_idx + 1
    end_human = last_completed + 1
    final_name = f"{SEC_CLIMATE_PREFIX}_{start_human:05d}_{end_human:05d}.csv.gz"
    final_path = CLEAN_DATA_DIR / final_name

    if final_path.exists():
        ts = int(time.time())
        final_path = CLEAN_DATA_DIR / f"{SEC_CLIMATE_PREFIX}_{start_human:05d}_{end_human:05d}_{ts}.csv.gz"
        print(f"[WARN] Final file existed; using {final_path.name}")

    temp_path.rename(final_path)
    print(f"[INFO] Saved chunk: {final_path}")


if __name__ == "__main__":
    main()
