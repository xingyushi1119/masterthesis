# masterthesis
# News Sentiment and the Cross-Section of Tech-Stock Returns

Master's thesis code. Tests whether daily news sentiment predicts
next-period excess returns for a universe of US technology stocks, after
controlling for standard risk factors.

**Headline finding:** once you control for the Fama–French 6 factors and firm
characteristics, the main sentiment effect is statistically **insignificant**
at both daily and weekly frequency, and this null holds across three
independent sentiment sources (RavenPack ESS, FinBERT, VADER). A daily
small-cap interaction is significant under RavenPack but does not replicate
under FinBERT, so it is not treated as a robust result.

---

## Data (not included — licensed)

This repo is **code only**. The inputs are under commercial / academic
license and cannot be redistributed:

| Source | What it provides | Where to get it |
|---|---|---|
| RavenPack | Event-level news sentiment (ESS), entity ids, CUSIP | RavenPack subscription |
| RavenPack headlines | Raw headlines (for FinBERT/VADER robustness) | RavenPack subscription |
| CRSP daily (via WRDS) | Returns, prices, the panel with FF6 + firm controls | WRDS / CRSP |

To run the pipeline, place the source files in `data/` (see paths in
`src/config.py`) or point `THESIS_DATA_DIR` at wherever they live:

```bash
export THESIS_DATA_DIR=/path/to/licensed/data
export THESIS_OUT_DIR=/path/to/outputs
```

Expected raw inputs: `ravenpack{2021..2024}.csv.zip`,
`final_with_beta_clean.csv` (CRSP panel), `permno_list_with_cusip.csv`.

---

## Method

- **Target:** `y = r_{i,t+1} − RF_t` (one-period-ahead excess return).
- **Estimator:** firm-demeaned OLS (≡ firm fixed effects) with **two-way
  clustered** standard errors (firm and date).
- **Controls:** FF6 (MKT_RF, SMB, HML, RMW, CMA, MOM) + size, turnover,
  252-day beta, 60-day volatility.
- **Point-in-time alignment:** news is timestamped in ET; anything at/after
  16:00 ET is pushed to the next session, then mapped forward to the nearest
  trading day — so no look-ahead.
- **Leave-one-out (LOO) sector index:** the firm's own sentiment is excluded
  when building its sector signal, to avoid mechanical self-correlation.
- **Robustness:** the entire sentiment pipeline is re-run with FinBERT
  (`ProsusAI/finbert`, prob(pos)−prob(neg) on the headline) and VADER
  (compound polarity) in place of RavenPack's ESS.

### A note on the weekly specification
Weekly `y` averages the daily t+1 excess returns within a week, which overlaps
the within-week factor realizations. This mechanically inflates the weekly
market-factor loading relative to daily. The daily spec (t vs t+1, no overlap)
is the cleaner one; weekly is reported as a secondary frequency.

---

## Pipeline

Run in order; each script reads paths from `src/config.py`.

| Step | Script | Output |
|---|---|---|
| 1 | `01_build_sentiment_indices.py` | filter → dedup → trading-day align → firm/sector/LOO daily indices |
| 2 | `02_merge_sentiment_to_panel.py` | sentiment merged onto the CRSP panel |
| 3 | `03_run_regressions.py` | daily + weekly regression tables in `outputs/` |
| 4 | `04_robustness_finbert.py` | FinBERT headline scores + rebuilt indices |
| 5 | `05_robustness_vader.py` | VADER headline scores + rebuilt indices |

```bash
pip install -r requirements.txt
cd src
python 01_build_sentiment_indices.py
python 02_merge_sentiment_to_panel.py
python 03_run_regressions.py
# robustness:
python 04_robustness_finbert.py
python 05_robustness_vader.py
```

---

## Scale (typical run)

~9.8M raw RavenPack rows → ~604k after filter/dedup → ~158k firm-days across
~681 tech CUSIPs, 2021–2024. Regression panel ~515k firm-day observations.
