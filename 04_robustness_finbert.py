"""
Step 4 (robustness) - Replace RavenPack's event_sentiment_score with a
FinBERT score computed from the news headline, then rebuild the firm-day /
sector-day / LOO indices and re-run the baseline daily regressions.

Purpose: check that the (null) main result is not an artifact of RavenPack's
proprietary sentiment scoring. The headline-level FinBERT score is
prob(positive) - prob(negative) from ProsusAI/finbert.

Requires the headline file (licensed, joined to RavenPack story/entity ids)
and the same CRSP panel as the main pipeline.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config as cfg

HEADLINES = cfg.DATA_DIR / "rp_aligned_with_headline.csv.gz"
EVENT_FINBERT = cfg.DATA_DIR / "event_with_finbert.csv.gz"
FB_DIR = cfg.DATA_DIR / "Finbert_firm_day"
FB_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Score headlines with FinBERT
# ---------------------------------------------------------------------------
def score_headlines():
    df = pd.read_csv(
        HEADLINES,
        usecols=["rp_story_id", "rp_entity_id", "cusip8",
                 "trading_session_date", "headline"],
        dtype="string", compression="infer")
    df["headline"] = df["headline"].fillna("")

    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()

    def finbert_score(texts, batch_size=64):
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=64, return_tensors="pt")
                probs = torch.softmax(model(**enc).logits, dim=1).cpu().numpy()
                scores.extend((probs[:, 2] - probs[:, 0]).tolist())  # pos - neg
        return scores

    df["sent_finbert"] = finbert_score(df["headline"].tolist())
    df.to_csv(EVENT_FINBERT, index=False, compression="gzip")
    print(f"[finbert] scored {len(df):,} headlines")
    return df


# ---------------------------------------------------------------------------
# 2. Rebuild firm-day / sector-day / LOO from the FinBERT score
# ---------------------------------------------------------------------------
def rebuild_indices():
    df = pd.read_csv(EVENT_FINBERT, low_memory=False, compression="infer")
    df["trading_session_date"] = (
        pd.to_datetime(df["trading_session_date"], errors="coerce").dt.normalize())
    df = df[df["sent_finbert"].notna() & df["cusip8"].ne("")]

    firm_day = (df.groupby(["cusip8", "trading_session_date"], observed=True)
                .agg(sent_firm_finbert=("sent_finbert", "mean"),
                     n_headlines=("sent_finbert", "size"))
                .reset_index())

    sector_day = (firm_day.groupby("trading_session_date", observed=True)
                  .agg(sent_index_finbert=("sent_firm_finbert", "mean"),
                       n_firms=("cusip8", "nunique"))
                  .reset_index())
    sector_day.loc[sector_day["n_firms"] < cfg.BREADTH_MIN,
                   "sent_index_finbert"] = np.nan
    qlo, qhi = sector_day["sent_index_finbert"].quantile(cfg.WINSOR_P)
    sector_day["sent_index_finbert"] = sector_day["sent_index_finbert"].clip(qlo, qhi)

    sums = (firm_day.groupby("trading_session_date", observed=True)
            .agg(sum_sent=("sent_firm_finbert", "sum"),
                 n_firms=("cusip8", "nunique"))
            .reset_index())
    fd = firm_day.merge(sums, on="trading_session_date", how="left")
    denom = (fd["n_firms"] - 1).replace(0, np.nan)
    fd["sent_sector_loo_finbert"] = (fd["sum_sent"] - fd["sent_firm_finbert"]) / denom
    loo = fd[["cusip8", "trading_session_date", "sent_sector_loo_finbert"]]

    firm_day.to_csv(FB_DIR / "finbert_firm_day_sentiment.csv", index=False)
    sector_day.to_csv(FB_DIR / "finbert_sector_day_sentiment.csv", index=False)
    loo.to_csv(FB_DIR / "finbert_firm_day_LOO.csv", index=False)
    print(f"[finbert] firm-day {firm_day.shape}")


if __name__ == "__main__":
    score_headlines()
    rebuild_indices()
    # The rebuilt FinBERT indices are merged onto the panel (same logic as
    # step 2) and fed through run_demeaned() from step 3 to reproduce the
    # baseline + size-heterogeneity specs under FinBERT sentiment.
