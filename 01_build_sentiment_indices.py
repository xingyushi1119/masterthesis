"""
Step 1 - Build daily news-sentiment indices from RavenPack.

Pipeline:
  1. Filter each yearly RavenPack file to US equities in the CUSIP universe,
     non-missing event_sentiment_score (ESS), within the sample dates.
  2. Concatenate years and deduplicate on (rp_story_id, rp_entity_id),
     preferring US ISIN, then higher relevance / event_relevance.
  3. Align each event to a trading day: convert to ET, push >=16:00 ET news to
     the next session, then map to the nearest forward trading day
     (point-in-time, so no look-ahead).
  4. Aggregate events -> firm-day (equal weight), firm-day -> sector-day
     (breadth filter + winsorize), and build a leave-one-out (LOO) sector
     index that excludes the firm itself (avoids mechanical self-correlation).

Source data is licensed (RavenPack, CRSP) and is NOT included in the repo.
"""

import zipfile

import numpy as np
import pandas as pd

import config as cfg


def norm_cusip8(s: pd.Series) -> pd.Series:
    """Uppercase, strip non-alphanumerics, keep last 8 chars, left-pad to 8."""
    s = s.astype(str).str.upper().str.replace(r"[^0-9A-Z]", "", regex=True)
    return s.str[-8:].str.zfill(8)


# ---------------------------------------------------------------------------
# 1. Filter each yearly RavenPack file
# ---------------------------------------------------------------------------
def filter_year(zip_path, out_path, cusip8_set, start, end):
    usecols = [
        "rp_story_id", "rp_entity_id", "entity_type", "entity_name",
        "country_code", "relevance", "event_relevance", "event_sentiment_score",
        "isin", "cusip", "rpa_date_utc", "rpa_time_utc", "timestamp_utc",
    ]
    first_chunk = True
    with zipfile.ZipFile(zip_path, "r") as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f:
            for chunk in pd.read_csv(f, encoding="latin1", usecols=usecols,
                                     chunksize=300_000, low_memory=False):
                chunk["rpa_date_utc"] = pd.to_datetime(chunk["rpa_date_utc"],
                                                       errors="coerce")
                mask = (
                    chunk["event_sentiment_score"].notna()
                    & (chunk["country_code"] == "US")
                    & (chunk["rpa_date_utc"] >= start)
                    & (chunk["rpa_date_utc"] <= end)
                )
                sub = chunk.loc[mask].copy()
                sub["cusip8"] = sub["cusip"].astype(str).str[:8]
                sub = sub[sub["cusip8"].isin(cusip8_set)]
                sub.to_csv(out_path, index=False,
                           mode="wt" if first_chunk else "at",
                           header=first_chunk, compression="gzip")
                first_chunk = False
    print(f"[filter] {zip_path.name} -> {out_path.name}")


# ---------------------------------------------------------------------------
# 2. Concatenate + deduplicate
# ---------------------------------------------------------------------------
def dedup(filtered_paths, out_path):
    dtypes = {
        "rp_story_id": "string", "rp_entity_id": "string",
        "entity_type": "string", "entity_name": "string",
        "country_code": "string", "relevance": "float64",
        "event_relevance": "float64", "event_sentiment_score": "float64",
        "isin": "string", "cusip": "string",
        "rpa_date_utc": "string", "rpa_time_utc": "string",
        "timestamp_utc": "string",
    }
    raw = pd.concat([pd.read_csv(p, dtype=dtypes, low_memory=False)
                     for p in filtered_paths], ignore_index=True)
    raw["rpa_date_utc"] = pd.to_datetime(raw["rpa_date_utc"], errors="coerce")

    # Prefer US ISIN, then higher relevance, then higher event_relevance.
    us_first = (~raw["isin"].fillna("").str.startswith("US")).astype(int)
    raw["relevance"] = pd.to_numeric(raw["relevance"], errors="coerce")
    raw["event_relevance"] = pd.to_numeric(raw["event_relevance"], errors="coerce")
    ordered = raw.assign(_us_first=us_first).sort_values(
        by=["rp_story_id", "rp_entity_id", "_us_first", "relevance", "event_relevance"],
        ascending=[True, True, True, False, False],
    )
    clean = ordered.drop_duplicates(subset=["rp_story_id", "rp_entity_id"],
                                    keep="first")
    clean = clean.sort_values(["rpa_date_utc", "rp_story_id", "rp_entity_id"])
    clean = clean.drop(columns="_us_first")
    clean.to_csv(out_path, index=False, compression="gzip")
    print(f"[dedup] {len(raw):,} -> {len(clean):,} rows")
    return clean


# ---------------------------------------------------------------------------
# 3. Trading-day calendar + point-in-time alignment
# ---------------------------------------------------------------------------
def build_trading_days(dsf_path, out_path, since="2020-07-01"):
    df = pd.read_csv(dsf_path)
    df["date"] = pd.to_datetime(df["date"])
    days = (df.loc[df["date"] >= since, "date"]
            .drop_duplicates().sort_values())
    days.to_frame("trading_date").to_csv(out_path, index=False)
    print(f"[calendar] {len(days)} trading days")
    return days


def _build_ts_utc(df):
    if "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"].astype(str).str.strip(),
                            errors="coerce", utc=True)
    else:
        ts = pd.Series(pd.NaT, index=df.index)
    need = ts.isna()
    if need.any():
        d = df.loc[need, "rpa_date_utc"].astype(str).str.strip()
        t = (df.loc[need, "rpa_time_utc"].astype(str).str.strip()
             .str.replace(",", ".", regex=False))
        is_24 = t.str.startswith("24:")
        d2 = d.where(~is_24, (pd.to_datetime(d, errors="coerce")
                              + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"))
        t2 = t.where(~is_24, t.str.replace("24:", "00:", regex=False))
        ts2 = pd.to_datetime(d2 + " " + t2, errors="coerce", utc=True)
        ts.loc[need] = ts2.values
    return pd.to_datetime(ts, errors="coerce", utc=True)


def align_to_trading_day(dedup_path, calendar_path, out_path, drop_oob=True):
    news = pd.read_csv(dedup_path, low_memory=False)
    cal = pd.read_csv(calendar_path)
    cal_idx = pd.DatetimeIndex(
        pd.to_datetime(cal["trading_date"]).dt.normalize().drop_duplicates().sort_values()
    )

    news["ts_utc"] = _build_ts_utc(news)
    ts_et = news["ts_utc"].dt.tz_convert("America/New_York")
    is_after_cut = ts_et.dt.hour >= cfg.CUTOFF_HOUR
    session = ts_et.dt.normalize().dt.tz_localize(None)
    session = session.where(~is_after_cut, session + pd.Timedelta(days=1))

    work = (pd.DataFrame({"left_key": session})
            .sort_values("left_key"))
    cal_df = pd.DataFrame({"right_key": cal_idx}).sort_values("right_key")
    aligned = pd.merge_asof(work, cal_df, left_on="left_key",
                            right_on="right_key", direction="forward")
    news["trading_session_date"] = aligned["right_key"].values

    oob = news["ts_utc"].notna() & news["trading_session_date"].isna()
    if drop_oob and oob.any():
        print(f"[align] dropping {int(oob.sum())} out-of-range events")
        news = news.loc[~oob].copy()
    news.to_csv(out_path, index=False, compression="gzip")
    print(f"[align] {len(news):,} events aligned -> {out_path.name}")
    return news


# ---------------------------------------------------------------------------
# 4. Aggregate to firm-day / sector-day / LOO
# ---------------------------------------------------------------------------
def aggregate(aligned_path):
    ID, DATE, ESS, RELEV = "cusip8", "trading_session_date", \
        "event_sentiment_score", "event_relevance"

    df = pd.read_csv(aligned_path, low_memory=False, compression="infer")
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce").dt.normalize()
    df[ID] = norm_cusip8(df[ID])
    df = df[df[DATE].notna() & df[ESS].notna() & df[ID].ne("")]
    df[RELEV] = pd.to_numeric(df[RELEV], errors="coerce")
    df = df[df[RELEV] >= cfg.RELEV_CUTOFF]

    df["_pos"] = np.where(df[ESS] > 0, df[ESS], 0.0)
    df["_neg"] = np.where(df[ESS] < 0, df[ESS], 0.0)

    firm_day = (df.groupby([ID, DATE], observed=True)
                .agg(sent_firm=(ESS, "mean"),
                     pos_firm=("_pos", "mean"),
                     neg_firm=("_neg", "mean"),
                     n_events=(ESS, "size"))
                .reset_index())

    sector_day = (firm_day.groupby(DATE, observed=True)
                  .agg(sent_index=("sent_firm", "mean"),
                       pos_index=("pos_firm", "mean"),
                       neg_index=("neg_firm", "mean"),
                       n_firms=(ID, "nunique"))
                  .reset_index())
    low = sector_day["n_firms"] < cfg.BREADTH_MIN
    sector_day.loc[low, ["sent_index", "pos_index", "neg_index"]] = np.nan
    for c in ["sent_index", "pos_index", "neg_index"]:
        qlo, qhi = sector_day[c].quantile(cfg.WINSOR_P)
        sector_day[c] = sector_day[c].clip(qlo, qhi)

    # Leave-one-out sector index, built from firm_day (not the truncated index).
    sums = (firm_day.groupby(DATE, observed=True)
            .agg(sum_sent=("sent_firm", "sum"),
                 sum_pos=("pos_firm", "sum"),
                 sum_neg=("neg_firm", "sum"),
                 n_firms=(ID, "nunique"))
            .reset_index())
    fd = firm_day.merge(sums, on=DATE, how="left")
    denom = (fd["n_firms"] - 1).replace(0, np.nan)
    fd["sent_sector_loo"] = (fd["sum_sent"] - fd["sent_firm"]) / denom
    fd["pos_sector_loo"] = (fd["sum_pos"] - fd["pos_firm"]) / denom
    fd["neg_sector_loo"] = (fd["sum_neg"] - fd["neg_firm"]) / denom
    loo = fd[[ID, DATE, "sent_sector_loo", "pos_sector_loo", "neg_sector_loo"]]

    firm_day.to_csv(cfg.FIRM_DAY, index=False)
    sector_day.to_csv(cfg.SECTOR_DAY, index=False)
    loo.to_csv(cfg.LOO_FIRM_DAY, index=False)
    print(f"[aggregate] firm-day {firm_day.shape}, sector-day {sector_day.shape}")
    return firm_day, sector_day, loo


def main():
    cusip_df = pd.read_csv(cfg.PERMNO_CUSIP_MAP)
    cusip8_set = set(cusip_df["CUSIP"].astype(str).str[:8].unique())
    print(f"CUSIP universe: {len(cusip8_set)}")

    start = pd.Timestamp(cfg.SAMPLE_START)
    end = pd.Timestamp(cfg.SAMPLE_END)
    for yr, zip_path in cfg.RAVENPACK_ZIPS.items():
        filter_year(zip_path, cfg.RP_FILTERED[yr], cusip8_set, start, end)

    dedup(list(cfg.RP_FILTERED.values()), cfg.RP_DEDUP)
    build_trading_days(cfg.DSF_PANEL, cfg.TRADING_DAYS)
    align_to_trading_day(cfg.RP_DEDUP, cfg.TRADING_DAYS, cfg.RP_ALIGNED)
    aggregate(cfg.RP_ALIGNED)


if __name__ == "__main__":
    main()
