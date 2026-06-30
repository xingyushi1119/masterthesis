"""
Step 5 (robustness) - Same as step 4 but with VADER (a lexicon-based scorer)
instead of FinBERT, as a second independent check on the RavenPack result.
The VADER score is the 'compound' polarity of the headline.
"""

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import config as cfg

EVENT_FINBERT = cfg.DATA_DIR / "event_with_finbert.csv.gz"
EVENT_VADER = cfg.DATA_DIR / "event_with_finbert_vader.csv.gz"
V_DIR = cfg.DATA_DIR / "Vader_firm_day"
V_DIR.mkdir(exist_ok=True)


def score_headlines():
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

    df = pd.read_csv(EVENT_FINBERT, low_memory=False)
    df["trading_session_date"] = pd.to_datetime(df["trading_session_date"])
    df["sent_vader"] = df["headline"].fillna("").apply(
        lambda x: sia.polarity_scores(x)["compound"])
    df.to_csv(EVENT_VADER, index=False, compression="gzip")
    print(f"[vader] scored {len(df):,} headlines")
    return df


def rebuild_indices():
    df = pd.read_csv(EVENT_VADER, low_memory=False)
    df["trading_session_date"] = (
        pd.to_datetime(df["trading_session_date"]).dt.normalize())
    df = df[df["sent_vader"].notna()]

    firm_day = (df.groupby(["cusip8", "trading_session_date"], observed=True)
                .agg(sent_firm_vader=("sent_vader", "mean"),
                     n_headlines=("sent_vader", "size"))
                .reset_index())

    sector_day = (firm_day.groupby("trading_session_date", observed=True)
                  .agg(sent_index_vader=("sent_firm_vader", "mean"),
                       n_firms=("cusip8", "nunique"))
                  .reset_index())

    sums = (firm_day.groupby("trading_session_date", observed=True)
            .agg(sum_sent=("sent_firm_vader", "sum"),
                 n_firms=("cusip8", "nunique"))
            .reset_index())
    fd = firm_day.merge(sums, on="trading_session_date", how="left")
    denom = (fd["n_firms"] - 1).replace(0, np.nan)
    fd["sent_sector_loo_vader"] = (fd["sum_sent"] - fd["sent_firm_vader"]) / denom
    loo = fd[["cusip8", "trading_session_date", "sent_sector_loo_vader"]]

    firm_day.to_csv(V_DIR / "vader_firm_day_sentiment.csv", index=False)
    sector_day.to_csv(V_DIR / "vader_sector_day_sentiment.csv", index=False)
    loo.to_csv(V_DIR / "vader_firm_day_LOO.csv", index=False)
    print(f"[vader] firm-day {firm_day.shape}")


if __name__ == "__main__":
    score_headlines()
    rebuild_indices()
    # As with FinBERT, the rebuilt VADER indices are merged onto the panel and
    # run through run_demeaned() from step 3 for the baseline daily specs.
