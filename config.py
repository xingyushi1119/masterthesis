"""
Central configuration: all paths and shared constants live here.

The original notebooks hard-coded absolute paths on two different machines
(macOS and Windows). Everything is now relative to DATA_DIR / OUT_DIR, which
default to ../data and ../outputs but can be overridden with environment
variables so the code runs on any machine that has the (licensed) source data.

    export THESIS_DATA_DIR=/path/to/your/data
    export THESIS_OUT_DIR=/path/to/your/outputs
"""

import os
from pathlib import Path

# ----- Base directories -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("THESIS_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(os.environ.get("THESIS_OUT_DIR", ROOT / "outputs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Raw / external inputs (NOT in the repo - see README) -------------------
# RavenPack annual files, one zip per year, e.g. ravenpack2021.csv.zip
RAVENPACK_ZIPS = {
    2021: DATA_DIR / "ravenpack2021.csv.zip",
    2022: DATA_DIR / "ravenpack2022.csv.zip",
    2023: DATA_DIR / "ravenpack2023.csv.zip",
    2024: DATA_DIR / "ravenpack2024.csv.zip",
}
# CRSP daily stock file panel (with FF6 factors + firm controls already merged)
DSF_PANEL = DATA_DIR / "final_with_beta_clean.csv"
# PERMNO <-> CUSIP mapping (one row per security)
PERMNO_CUSIP_MAP = DATA_DIR / "permno_list_with_cusip.csv"

# ----- Intermediate artifacts (produced by the pipeline) ----------------------
RP_FILTERED = {yr: DATA_DIR / f"rp_filtered_cusip8_{yr}.csv.gz" for yr in RAVENPACK_ZIPS}
RP_DEDUP = DATA_DIR / "rp_filtered_cusip8_2021_2024_merged_dedup.csv.gz"
TRADING_DAYS = DATA_DIR / "trading_days_20200701_onward.csv"
RP_ALIGNED = DATA_DIR / "rp_filtered_cusip8_2021_2024_aligned.csv.gz"

FIRM_DAY = DATA_DIR / "rp_firm_day_sentiment_eq_cusip8.csv"
SECTOR_DAY = DATA_DIR / "rp_sector_day_sentiment_eq_cusip8.csv"
LOO_FIRM_DAY = DATA_DIR / "rp_firm_day_LOO_eq_cusip8.csv"

PANEL_WITH_SENTIMENT = DATA_DIR / "final_with_beta_clean_plus_sentiment.csv"

# ----- Column / model constants ----------------------------------------------
FIRM, DATE = "PERMNO", "date"
RET, RF = "ret_adj", "RF"
FF6 = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
CTRLS = ["size", "turnover", "beta_252d", "vol_60d"]

# Sample window for the regressions
SAMPLE_START = "2021-01-01"
SAMPLE_END = "2024-12-31"

# Aggregation parameters
RELEV_CUTOFF = 50          # minimum event_relevance to keep an event
BREADTH_MIN = 5            # minimum #firms for a valid sector-day index
WINSOR_P = (0.01, 0.99)    # winsorization quantiles
CUTOFF_HOUR = 16           # 16:00 ET: news after this prints to next trading day
