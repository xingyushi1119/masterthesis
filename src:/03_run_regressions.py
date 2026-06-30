"""
Step 3 - Predictive regressions: does news sentiment forecast next-period
excess return, after controlling for FF6 factors and firm characteristics?

Target:   y = r_{i,t+1} - RF_t   (one-period-ahead excess return)
Model:    firm-demeaned OLS (equivalent to firm fixed effects) with
          two-way clustered standard errors (firm and date).
Controls: FF6 (MKT_RF, SMB, HML, RMW, CMA, MOM) + size, turnover,
          beta_252d, vol_60d.

Specifications, run at both DAILY and WEEKLY frequency:
  - net sentiment (loo_net / sector_net), raw and cross-sectionally z-scored
  - positive / negative components (pos, neg)
  - heterogeneity: interactions with small / highvol / highbeta / highturn
    median splits, plus a continuous size-z interaction.

Reading the results: the main net-sentiment effect is statistically
insignificant at both frequencies (the headline finding is a null). See README.

Caveat on WEEKLY: weekly y is the average of t+1 daily excess returns within
the week, which mechanically overlaps the within-week factor realizations, so
the weekly market-factor loading is inflated relative to daily. Daily uses
t vs t+1 with no overlap and is the cleaner specification.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------
def run_demeaned(df_in, firm, date, ycol, main_xcols, tag, out_dir):
    """Firm-demeaned OLS with (firm, date) two-way clustered SE."""
    xcols = [c for c in main_xcols + cfg.FF6 + cfg.CTRLS if c in df_in.columns]
    d = df_in[[firm, date, ycol] + xcols].copy()
    for c in xcols + [ycol]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    if d.empty:
        print(f"[skip] {tag}: empty sample")
        return None

    def dm(g):
        return g - g.mean()

    d["y_dm"] = d.groupby(firm, group_keys=False)[ycol].transform(dm)
    X_dm = pd.DataFrame({c: d.groupby(firm, group_keys=False)[c].transform(dm)
                         for c in xcols})

    # Drop columns that are constant or perfectly collinear after demeaning.
    zero = [c for c in X_dm.columns
            if np.isclose(X_dm[c].std(skipna=True), 0.0)]
    X_dm = X_dm.drop(columns=zero)
    if X_dm.shape[1] >= 2:
        corr = X_dm.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop2 = [c for c in upper.columns if (upper[c] > 0.999999).any()]
        X_dm = X_dm.drop(columns=drop2)

    groups = np.c_[d[firm].values, pd.factorize(d[date].values)[0]]
    res = sm.OLS(d["y_dm"].values, X_dm.values).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    out = pd.DataFrame({"var": X_dm.columns, "coef": res.params,
                        "se": res.bse, "t": res.tvalues, "pval": res.pvalues})
    out.to_csv(out_dir / f"core_model_full_{tag}.csv", index=False)
    print(f"[ok] {tag}")
    return out


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def add_dummies(panel, time_col):
    splits = {"small": ("size", "<="), "highvol": ("vol_60d", ">="),
              "highbeta": ("beta_252d", ">="), "highturn": ("turnover", ">=")}
    for flag, (col, op) in splits.items():
        if col in panel.columns:
            med = panel.groupby(time_col)[col].transform("median")
            panel[flag] = ((panel[col] <= med) if op == "<="
                           else (panel[col] >= med)).astype(int)
    return panel


def add_interactions(panel, net_cols,
                     flags=("small", "highvol", "highbeta", "highturn")):
    for v in net_cols:
        for f in flags:
            if v in panel.columns and f in panel.columns:
                panel[f"{v}_x_{f}"] = panel[v] * panel[f]
    return panel


def xsec_z(panel, time_col, col):
    mu = panel.groupby(time_col)[col].transform("mean")
    sd = panel.groupby(time_col)[col].transform("std").replace(0, np.nan)
    return (panel[col].astype(float) - mu) / sd


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------
def build_specs(panel, net_cols, size_z_col):
    specs = [("loo_posneg", ["loo_pos", "loo_neg"]),
             ("sector_posneg", ["sector_pos", "sector_neg"])]
    for nc in net_cols:
        specs.append((nc, [nc]))
        if f"{nc}_z" in panel.columns:
            specs.append((f"{nc}_z", [f"{nc}_z"]))
        for flag in ("small", "highvol", "highbeta", "highturn"):
            need = [nc, flag, f"{nc}_x_{flag}"]
            if all(c in panel.columns for c in need):
                specs.append((f"hetero_{nc}_{flag}", need))
        cont = [nc, size_z_col, f"{nc}_x_{size_z_col}"]
        if all(c in panel.columns for c in cont):
            specs.append((f"hetero_{nc}_x_{size_z_col}", cont))
    return specs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(cfg.PANEL_WITH_SENTIMENT, low_memory=False)
    df[cfg.DATE] = pd.to_datetime(df[cfg.DATE], errors="coerce")
    df = df[(df[cfg.DATE] >= cfg.SAMPLE_START)
            & (df[cfg.DATE] <= cfg.SAMPLE_END)].sort_values([cfg.FIRM, cfg.DATE])
    df["ret_lead"] = df.groupby(cfg.FIRM, group_keys=False)[cfg.RET].shift(-1)
    df["y"] = df["ret_lead"] - df[cfg.RF]

    # ---------------- DAILY ----------------
    net_daily = [c for c in ["loo_net", "sector_net"] if c in df.columns]
    for c in net_daily:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
    df = add_dummies(df, cfg.DATE)
    df = add_interactions(df, net_daily)
    if "size" in df.columns:
        df["size_z_day"] = xsec_z(df, cfg.DATE, "size")
        for c in net_daily:
            df[f"{c}_x_size_z_day"] = df[c] * df["size_z_day"]

    for tag, xs in build_specs(df, net_daily, "size_z_day"):
        if all(c in df.columns for c in xs):
            run_demeaned(df, cfg.FIRM, cfg.DATE, "y", xs, tag, cfg.OUT_DIR)

    # ---------------- WEEKLY ----------------
    df["week"] = df[cfg.DATE].dt.to_period("W").apply(lambda r: r.start_time)
    agg = {"y": "mean", "size": "last", "turnover": "mean",
           "beta_252d": "last", "vol_60d": "mean",
           **{f: "mean" for f in cfg.FF6}}
    for c in ["loo_net", "sector_net", "loo_pos", "loo_neg",
              "sector_pos", "sector_neg"]:
        if c in df.columns:
            agg[c] = "mean"
    weekly = df.groupby([cfg.FIRM, "week"]).agg(agg).reset_index()

    net_weekly = [c for c in ["loo_net", "sector_net"] if c in weekly.columns]
    for c in net_weekly:
        weekly[f"{c}_z"] = (weekly[c] - weekly[c].mean()) / weekly[c].std(ddof=0)
    weekly = add_dummies(weekly, "week")
    weekly = add_interactions(weekly, net_weekly)
    if "size" in weekly.columns:
        weekly["size_z_week"] = xsec_z(weekly, "week", "size")
        for c in net_weekly:
            weekly[f"{c}_x_size_z_week"] = weekly[c] * weekly["size_z_week"]

    for tag, xs in build_specs(weekly, net_weekly, "size_z_week"):
        if all(c in weekly.columns for c in xs):
            run_demeaned(weekly, cfg.FIRM, "week", "y", xs,
                         f"weekly_{tag}", cfg.OUT_DIR)

    print("[done] daily + weekly regressions complete")


if __name__ == "__main__":
    main()
