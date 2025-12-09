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

# Small sleep between filings to be polite to SEC
SLEEP_BETWEEN_FILINGS_SEC = float(os.environ.get("SLEEP_BETWEEN_FILINGS_SEC", "0.2"))

# Optional external keyword file (one term per line)
CLIMATE_KEYWORD_PATH = os.environ.get(
    "CLIMATE_KEYWORD_PATH",
    "data/keywords/climate_keywords_base.txt",
)

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


# ========== BASIC HELPERS ==========

def ensure_dir(path: str) -> None:
    """Create parent directory if it does not exist."""
    p = Path(pat
