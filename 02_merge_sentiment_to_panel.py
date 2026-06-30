"""
Step 2 - Merge the daily sentiment indices onto the CRSP daily panel.

  - sector index joins on date (one value per trading day)
  - LOO index joins on (PERMNO, date) via the CUSIP8 -> PERMNO mapping
  - on no-news days the firm's LOO value is backfilled with the sector value

NOTE: the original notebook had several malformed paths (missing path
separators, and the output path did not match what the regression step later
read). Paths are now centralized in config.py, so the output of this step is
exactly the file the regression step consumes (cfg.PANEL_WITH_SENTIMENT).
"""

import pandas as pd

import config as cfg
from importlib import import_module

norm_cusip8 = import_module("01_build_sentiment_indices").norm_cusip8


def main():
    # 1. CRSP daily panel
    dsf = pd.read_csv(cfg.DSF_PANEL, low_memory=False)
    dsf["date"] = pd.to_datetime(dsf["date"]).dt.normalize()

    # 2. Sector daily index -> join on date
    sector = pd.read_csv(cfg.SECTOR_DAY)
    sector["trading_session_date"] = (
        pd.to_datetime(sector["trading_session_date"]).dt.normalize())
    sector = sector.rename(columns={
        "trading_session_date": "date",
        "pos_index": "sector_pos",
        "neg_index": "sector_neg",
        "sent_index": "sector_net",
    })
    dsf = dsf.merge(sector[["date", "sector_pos", "sector_neg", "sector_net"]],
                    on="date", how="left")

    # 3. LOO firm-day index -> map CUSIP8 to PERMNO, then join on (PERMNO, date)
    loo = pd.read_csv(cfg.LOO_FIRM_DAY)
    loo["trading_session_date"] = (
        pd.to_datetime(loo["trading_session_date"]).dt.normalize())
    loo["cusip8"] = norm_cusip8(loo["cusip8"])

    m = pd.read_csv(cfg.PERMNO_CUSIP_MAP)
    if "cusip8" not in m.columns:
        if "CUSIP8" in m.columns:
            m = m.rename(columns={"CUSIP8": "cusip8"})
        elif "CUSIP" in m.columns:
            m["cusip8"] = norm_cusip8(m["CUSIP"])
    m["cusip8"] = norm_cusip8(m["cusip8"])
    m = m[["PERMNO", "cusip8"]].drop_duplicates()

    loo_perm = (loo.merge(m, on="cusip8", how="left")
                .dropna(subset=["PERMNO"])
                .rename(columns={"trading_session_date": "date"})
                .sort_values(["PERMNO", "date"])
                .drop_duplicates(["PERMNO", "date"], keep="first"))

    dsf = dsf.merge(
        loo_perm[["PERMNO", "date", "pos_sector_loo",
                  "neg_sector_loo", "sent_sector_loo"]],
        on=["PERMNO", "date"], how="left",
    )

    # 4. No-news day: firm LOO falls back to that day's sector index
    dsf["loo_pos"] = dsf["pos_sector_loo"].fillna(dsf["sector_pos"])
    dsf["loo_neg"] = dsf["neg_sector_loo"].fillna(dsf["sector_neg"])
    dsf["loo_net"] = dsf["sent_sector_loo"].fillna(dsf["sector_net"])
    dsf = dsf.drop(columns=["pos_sector_loo", "neg_sector_loo", "sent_sector_loo"])

    dsf.to_csv(cfg.PANEL_WITH_SENTIMENT, index=False)
    print(f"[merge] {dsf.shape} -> {cfg.PANEL_WITH_SENTIMENT.name}")


if __name__ == "__main__":
    main()
