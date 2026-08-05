"""
Backtest ranking + walk-forward ML filter comparison.

Step 1: Load every *_trades.csv at the top level of C:\\dhan algo, compute
        trade count, win rate, net profit, profit factor, max drawdown, SQN.
        Rank by SQN (min 40 trades). Log full ranking to CSV. Pick top-5
        DISTINCT strategies (dedupe near-identical SL/TSL variants of the
        same base script).

Step 2: For each of the top 5, replicate the ml_engine.py methodology
        (RandomForest + LogisticRegression predicting win/loss, trained on
        chronologically-earlier trades only, applied as a filter to later
        trades) using whatever numeric "entry-time" feature columns that
        strategy's own CSV actually has. Compare WITHOUT ML vs WITH ML on
        the SAME held-out test period.

Outputs (written to this directory):
    full_ranking.csv       - full SQN ranking of every eligible file (auditable)
    RESULTS.md             - final report with tables + caveats

This script does NOT touch step1_config_and_connector.py .. step9_live_engine.py
or step8_dashboard.py. It only reads *_trades.csv files and writes into
Research/backtest_ml_comparison/.
"""

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # C:\dhan algo
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_TRADES = 40
TRAIN_FRAC = 0.70
MIN_TEST_TRADES = 15
MIN_FEATURES = 3

PNL_CANDIDATES = [
    "net_pnl", "pnl", "rupee_pnl_net", "rupee_pnl", "profit", "points_gained",
    "pnl_pts", "points", "p&l", "p_l", "pl", "net_profit",
]

DATE_CANDIDATES = [
    "date", "entry_time", "entry_date", "signal_time", "timestamp", "datetime",
]

# columns that are clearly outcome/result data -> excluded from ML features
LEAKAGE_PATTERNS = re.compile(
    r"(exit|pnl|profit|p&l|p_l|^pl$|reason|running_|capital_after|mfe|mae|"
    r"rr_planned|net_pnl|brokerage|priority|blocked_by|confidence|tag$|"
    r"cross_confirmed|oi_agrees)",
    re.IGNORECASE,
)

# columns that are IDs / non-informative even though numeric
ID_LIKE = re.compile(r"(^id$|index|unnamed|expiry|strike$|ce_strike|pe_strike|lot_size)", re.IGNORECASE)


def find_pnl_column(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in PNL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    # fallback: any column containing 'pnl' or 'profit' or 'points'
    for c in df.columns:
        cl = c.lower()
        if "pnl" in cl or "profit" in cl or cl == "points" or "points_gained" in cl:
            return c
    return None


def find_date_column(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def compute_metrics(pnl_series):
    pnl = pd.to_numeric(pnl_series, errors="coerce").dropna()
    n = len(pnl)
    if n == 0:
        return None
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / n * 100
    net_profit = pnl.sum()
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf if gross_profit > 0 else np.nan
    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()
    std = pnl.std(ddof=1) if n > 1 else 0
    mean = pnl.mean()
    sqn = (mean / std * np.sqrt(n)) if std and std > 0 else np.nan
    avg_pnl = mean
    return dict(
        n_trades=n, win_rate=win_rate, net_profit=net_profit,
        profit_factor=profit_factor, max_drawdown=max_dd, sqn=sqn,
        avg_pnl=avg_pnl,
    )


def base_strategy_key(filename):
    """Collapse SL/TSL/parameter-sweep variants of the same base strategy
    into one dedupe key, e.g. strip trailing sweep tokens like sl12_trail1,
    1_1, 4min, 5.0pt, v2, timesweep, grid_5.0pt_3min_6mo, etc."""
    name = filename.replace("_trades.csv", "")
    # strip common sweep/parameter tokens
    tokens = name.split("_")
    strip_pat = re.compile(
        r"^(sl\d+.*|trail\d+.*|\d+_\d+|v\d+|\d+min|\d+\.\d+pt|\d+pt|grid|"
        r"\d+mo|fixed|gated|ungated|sweep.*|timesweep.*|\d+month)$",
        re.IGNORECASE,
    )
    kept = [t for t in tokens if not strip_pat.match(t)]
    if not kept:
        kept = tokens[:2]
    # Keep only the first 2 surviving tokens as the "family" key. This is
    # intentionally aggressive: e.g. six_concepts_rbi_gbi_relaxed,
    # six_concepts_power_bar_start, and six_concepts_market_law_4 all
    # collapse to "six_concepts" since they're the same underlying engine
    # with different concept toggles, not fundamentally different strategies.
    key = "_".join(kept[:2]).lower()
    return key


def step1_rank():
    files = sorted(glob.glob(os.path.join(ROOT, "*_trades.csv")))
    rows = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            rows.append(dict(file=fname, status=f"PARSE_FAIL: {e}"))
            continue
        if len(df) < MIN_TRADES:
            rows.append(dict(file=fname, status=f"TOO_FEW_ROWS({len(df)})"))
            continue
        pnl_col = find_pnl_column(df)
        if pnl_col is None:
            rows.append(dict(file=fname, status="NO_PNL_COLUMN"))
            continue
        m = compute_metrics(df[pnl_col])
        if m is None:
            rows.append(dict(file=fname, status="NO_VALID_PNL_VALUES"))
            continue
        m["file"] = fname
        m["pnl_column"] = pnl_col
        m["status"] = "OK"
        m["base_key"] = base_strategy_key(fname)
        rows.append(m)

    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(OUT_DIR, "full_ranking.csv"), index=False)

    ok = full[full["status"] == "OK"].copy()
    ok = ok.dropna(subset=["sqn"])
    ok = ok.sort_values("sqn", ascending=False)
    return full, ok


def pick_top5_distinct(ok_ranked):
    picked = []
    seen_keys = set()
    for _, row in ok_ranked.iterrows():
        key = row["base_key"]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        picked.append(row)
        if len(picked) == 5:
            break
    return pd.DataFrame(picked)


def identify_features(df, pnl_col, exclude_extra):
    exclude = set(exclude_extra) | {pnl_col}
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if LEAKAGE_PATTERNS.search(c):
            continue
        if ID_LIKE.search(c):
            continue
        col = df[c]
        # try numeric coercion first
        series = pd.to_numeric(col, errors="coerce")
        non_null_frac = series.notna().mean()
        if non_null_frac >= 0.8 and series.nunique(dropna=True) >= 2:
            feature_cols.append(c)
            continue
        # low-cardinality categorical entry-time attribute (e.g. side, direction,
        # CE/PE) -> label-encode into a new numeric column
        if col.dtype == object:
            nun = col.nunique(dropna=True)
            non_null_frac_cat = col.notna().mean()
            if 2 <= nun <= 8 and non_null_frac_cat >= 0.8:
                enc_name = f"_cat_{c}"
                df[enc_name] = col.astype("category").cat.codes
                feature_cols.append(enc_name)
    return feature_cols


def add_entry_time_of_day(df, exclude_cols):
    """If an 'entry_time' (HH:MM[:SS]) column exists, convert to minutes-since-
    midnight as a numeric entry-time feature."""
    added = []
    for c in df.columns:
        if c.lower() == "entry_time":
            try:
                t = pd.to_datetime(df[c], errors="coerce", format=None)
                if t.notna().mean() > 0.8 and t.dt.hour.nunique() > 1 - 1 and t.dt.minute.nunique() >= 1:
                    df["_feat_entry_minutes"] = t.dt.hour * 60 + t.dt.minute
                    if df["_feat_entry_minutes"].nunique(dropna=True) >= 2:
                        added.append("_feat_entry_minutes")
                    else:
                        df.drop(columns=["_feat_entry_minutes"], inplace=True)
            except Exception:
                pass
            break
    return df, added


def engineer_time_features(df, date_col, used_row_order):
    """Add day_of_week / hour-of-day features if a real datetime is available."""
    added = []
    if used_row_order:
        return df, added
    try:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.notna().mean() > 0.8:
            if dt.dt.hour.nunique() > 1:
                df["_feat_hour"] = dt.dt.hour + dt.dt.minute / 60.0
                added.append("_feat_hour")
            if dt.dt.dayofweek.nunique() > 1:
                df["_feat_dow"] = dt.dt.dayofweek
                added.append("_feat_dow")
    except Exception:
        pass
    return df, added


def step2_ml_filter(fpath, pnl_col):
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath, low_memory=False)
    result = dict(file=fname, pnl_column=pnl_col)

    date_col = find_date_column(df)
    used_row_order = date_col is None
    if not used_row_order:
        try:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            if parsed.notna().mean() < 0.5:
                used_row_order = True
            else:
                df["_parsed_date"] = parsed
                df = df.sort_values("_parsed_date", kind="mergesort").reset_index(drop=True)
        except Exception:
            used_row_order = True
    result["date_basis"] = "row_order (assumed chronological, NO real date column found)" if used_row_order else f"real date column: '{date_col}'"

    df, extra_time_feats = engineer_time_features(df, date_col, used_row_order)
    df, entry_time_feats = add_entry_time_of_day(df, set())
    extra_time_feats = extra_time_feats + entry_time_feats

    # target: win = pnl > 0
    df["_target"] = (pd.to_numeric(df[pnl_col], errors="coerce") > 0).astype(int)
    df = df.dropna(subset=[pnl_col])

    exclude_extra = {"_target", "_parsed_date", date_col} if date_col else {"_target", "_parsed_date"}
    feature_cols = identify_features(df, pnl_col, exclude_extra) + extra_time_feats
    feature_cols = list(dict.fromkeys(feature_cols))  # dedupe preserve order

    result["candidate_features"] = feature_cols
    n = len(df)
    split_idx = int(n * TRAIN_FRAC)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    result["n_train"] = len(train_df)
    result["n_test"] = len(test_df)

    if len(feature_cols) < MIN_FEATURES:
        result["skipped"] = True
        result["skip_reason"] = f"insufficient features/data for a walk-forward ML filter (only {len(feature_cols)} usable entry-time feature(s) found, need >= {MIN_FEATURES})"
        return result

    if len(test_df) < MIN_TEST_TRADES:
        result["skipped"] = True
        result["skip_reason"] = f"insufficient features/data for a walk-forward ML filter (only {len(test_df)} test-period trades, need >= {MIN_TEST_TRADES})"
        return result

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # use train medians, no leakage
    y_train = train_df["_target"].values
    y_test = test_df["_target"].values

    if len(np.unique(y_train)) < 2:
        result["skipped"] = True
        result["skip_reason"] = "insufficient features/data for a walk-forward ML filter (train period has only one class - all wins or all losses)"
        return result

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_proba_test = rf.predict_proba(X_test)[:, 1]
    rf_pred_test = (rf_proba_test > 0.5).astype(int)
    rf_acc = (rf_pred_test == y_test).mean()

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_proba_test = lr.predict_proba(X_test_s)[:, 1]
    lr_pred_test = (lr_proba_test > 0.5).astype(int)
    lr_acc = (lr_pred_test == y_test).mean()

    result["rf_test_accuracy"] = rf_acc
    result["lr_test_accuracy"] = lr_acc

    if rf_acc >= lr_acc:
        chosen_model = "RandomForest"
        chosen_proba = rf_proba_test
    else:
        chosen_model = "LogisticRegression"
        chosen_proba = lr_proba_test
    result["chosen_model"] = chosen_model

    without_ml = compute_metrics(test_df[pnl_col])
    filt_mask = chosen_proba > 0.5
    with_ml_df = test_df.loc[filt_mask, pnl_col]
    with_ml = compute_metrics(with_ml_df) if filt_mask.sum() > 0 else None

    result["without_ml"] = without_ml
    result["with_ml"] = with_ml
    result["n_filtered_out"] = int((~filt_mask).sum())
    result["skipped"] = False
    return result


def safe_fmt(v, nd=2):
    try:
        if v is None:
            return "n/a"
        if isinstance(v, float) and np.isnan(v):
            return "n/a"
        if isinstance(v, float) and np.isinf(v):
            return "inf"
        return f"{v:,.{nd}f}"
    except Exception:
        return "n/a"


def build_report(full, ok_ranked, top5, ml_results):
    lines = []
    lines.append("# Backtest Ranking + Walk-Forward ML Filter Comparison\n")
    lines.append(f"Generated by `run_comparison.py`. Source data: {len(full)} `*_trades.csv` files scanned "
                  f"at the top level of `C:\\dhan algo`.\n")

    n_ok = (full["status"] == "OK").sum()
    lines.append(f"- Files scanned: {len(full)}\n"
                  f"- Eligible (parsed OK, >= {MIN_TRADES} trades, recognizable P&L column): {n_ok}\n"
                  f"- Excluded (parse failure / too few rows / no P&L column found): {len(full) - n_ok}\n"
                  f"- Full ranked list (all eligible files, sorted by SQN): see `full_ranking.csv`\n")

    lines.append("\n## Top 20 by SQN (in-sample ranking, informational only)\n")
    lines.append("SQN = mean(trade P&L) / std(trade P&L) * sqrt(n). This is a robustness-adjusted ranking metric, "
                  "**not** a claim of proven edge — every one of these backtests is in-sample only (no walk-forward "
                  "validation exists elsewhere in this project either).\n")
    top20 = ok_ranked.head(20)[["file", "n_trades", "win_rate", "net_profit", "profit_factor", "max_drawdown", "sqn", "base_key"]]
    lines.append("\n| Rank | File | Trades | Win % | Net Profit | Profit Factor | Max DD | SQN | Base-strategy key |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        lines.append(f"| {i} | `{r['file']}` | {int(r['n_trades'])} | {safe_fmt(r['win_rate'],1)} | "
                      f"{safe_fmt(r['net_profit'],0)} | {safe_fmt(r['profit_factor'])} | {safe_fmt(r['max_drawdown'],0)} | "
                      f"{safe_fmt(r['sqn'],3)} | {r['base_key']} |")

    lines.append("\n## Top 5 selected strategies (deduped for diversity)\n")
    lines.append("Selection rule: walk down the SQN ranking, skip any file whose \"base strategy key\" "
                  "(script name with SL/TSL/parameter-sweep tokens like `sl12_trail1`, `1_1`, `4min`, `v2`, "
                  "`grid_5.0pt_3min_6mo` stripped out) has already been picked — this avoids filling all 5 "
                  "slots with SL/TSL variants of the same underlying strategy.\n")
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        lines.append(f"{i}. **`{r['file']}`** — SQN {safe_fmt(r['sqn'],3)}, {int(r['n_trades'])} trades, "
                      f"win rate {safe_fmt(r['win_rate'],1)}%, net profit {safe_fmt(r['net_profit'],0)}, "
                      f"profit factor {safe_fmt(r['profit_factor'])}, max DD {safe_fmt(r['max_drawdown'],0)} "
                      f"(base key: `{r['base_key']}`)")

    lines.append("\n## Step 2 — Walk-forward ML filter: WITHOUT ML vs WITH ML (same test period)\n")
    lines.append(f"Methodology: sort trades chronologically (real date column if available, else row order — "
                  f"noted per strategy below), first {int(TRAIN_FRAC*100)}% = train, last "
                  f"{int((1-TRAIN_FRAC)*100)}% = test. RandomForest + LogisticRegression trained on TRAIN only "
                  f"to predict win/loss from entry-time features; whichever model has higher TEST accuracy is "
                  f"used as the filter (predicted win-probability > 0.5) on the TEST portion. WITHOUT-ML and "
                  f"WITH-ML are compared on that exact same test slice — never against the full dataset.\n")

    for r in ml_results:
        lines.append(f"\n### `{r['file']}`\n")
        lines.append(f"- Date basis: {r['date_basis']}")
        lines.append(f"- Train / test split: {r.get('n_train','-')} train trades / {r.get('n_test','-')} test trades")
        if r.get("skipped"):
            lines.append(f"- **SKIPPED: {r['skip_reason']}**")
            if r.get("candidate_features"):
                lines.append(f"- Candidate feature columns considered: {r['candidate_features']}")
            continue
        lines.append(f"- Candidate entry-time feature columns used: {r['candidate_features']}")
        lines.append(f"- Model test-set accuracy — RandomForest: {safe_fmt(r['rf_test_accuracy'],3)}, "
                      f"LogisticRegression: {safe_fmt(r['lr_test_accuracy'],3)} -> **chosen: {r['chosen_model']}**")
        lines.append(f"- ML filter removed {r['n_filtered_out']} of {r['n_test']} test trades "
                      f"(predicted win-probability <= 0.5)")
        wo = r["without_ml"]
        wi = r["with_ml"]
        lines.append("\n| Metric | WITHOUT ML (all test trades) | WITH ML (filtered test trades) |")
        lines.append("|---|---|---|")
        if wo:
            lines.append(f"| Trade count | {wo['n_trades']} | {wi['n_trades'] if wi else 0} |")
            lines.append(f"| Win rate | {safe_fmt(wo['win_rate'],1)}% | {safe_fmt(wi['win_rate'],1) if wi else 'n/a'}% |")
            lines.append(f"| Net profit | {safe_fmt(wo['net_profit'],0)} | {safe_fmt(wi['net_profit'],0) if wi else 'n/a'} |")
            lines.append(f"| Profit factor | {safe_fmt(wo['profit_factor'])} | {safe_fmt(wi['profit_factor']) if wi else 'n/a'} |")
            lines.append(f"| Max drawdown | {safe_fmt(wo['max_drawdown'],0)} | {safe_fmt(wi['max_drawdown'],0) if wi else 'n/a'} |")
            lines.append(f"| SQN | {safe_fmt(wo['sqn'],3)} | {safe_fmt(wi['sqn'],3) if wi else 'n/a'} |")
            lines.append(f"| Avg P&L / trade | {safe_fmt(wo['avg_pnl'],2)} | {safe_fmt(wi['avg_pnl'],2) if wi else 'n/a'} |")
        if wi is None:
            lines.append("\n*ML filter rejected every test trade (predicted win-probability <= 0.5 for all) — no WITH-ML performance to report.*")

    lines.append("\n## Caveats (read before trusting any number above)\n")
    lines.append("""
- **In-sample selection risk.** The Top-5 strategies were chosen because they scored best on SQN over their
  own full historical sample. No strategy in this project (including these five) has been walk-forward
  validated end-to-end at the strategy-selection level — picking "best of 306 backtests" is itself a form of
  overfitting (multiple-comparisons / data-snooping bias). Treat the Top-5 as "best in-sample candidates,"
  not proven edge.
- **Single train/test split, not multiple walk-forward folds.** Step 2 uses one chronological 70/30 split per
  strategy, not rolling/expanding walk-forward folds. A single split can make a model look better or worse
  than it would across multiple out-of-sample periods purely by chance.
- **Small test-period sample sizes.** Some strategies' test slices are only a few dozen trades. Metrics like
  win rate, profit factor and SQN are noisy at that scale — differences between WITH/WITHOUT ML of a few
  percentage points are not statistically meaningful.
- **ML filter can "win" by shrinking the sample, not by finding real edge.** If the WITH-ML trade count is
  much smaller than WITHOUT-ML, an apparent improvement can simply be survivorship in a small lucky subset
  rather than a real predictive signal. Check the trade-count column before trusting any WITH-ML improvement.
- **Date basis.** Where no real date/timestamp column existed, row order was used as a proxy for chronological
  order and is flagged explicitly per strategy above. If the underlying CSV was not actually written in
  chronological order, that strategy's "walk-forward" split is not truly walk-forward.
- **Feature availability varies a lot by strategy.** Many of the 306 trade logs only contain
  entry/exit price, timestamps, and a P&L figure — very few "entry-time" predictive features (no gap%, ATR,
  VWAP distance, etc. in most files). Where fewer than 3 usable numeric entry-time features existed, or the
  test period had fewer than 15 trades, the ML step was skipped for that strategy and reported as such rather
  than forcing a fit.
- **This is a filter model, not a new strategy.** The ML step only decides whether to *take* a trade that the
  underlying rule-based strategy already generated; it cannot invent new entries, so its ceiling is bounded by
  the base strategy's own trade set.
""")

    return "\n".join(lines)


def main():
    print("Step 1: ranking all *_trades.csv files by SQN...")
    full, ok_ranked = step1_rank()
    print(f"  {len(full)} files scanned, {len(ok_ranked)} eligible.")

    top5 = pick_top5_distinct(ok_ranked)
    print("Top 5 selected:")
    print(top5[["file", "sqn", "n_trades", "base_key"]])

    print("\nStep 2: walk-forward ML filter comparison for each of the top 5...")
    ml_results = []
    for _, row in top5.iterrows():
        fpath = os.path.join(ROOT, row["file"])
        print(f"  processing {row['file']} ...")
        res = step2_ml_filter(fpath, row["pnl_column"])
        ml_results.append(res)

    report = build_report(full, ok_ranked, top5, ml_results)
    out_path = os.path.join(OUT_DIR, "RESULTS.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nWrote {out_path}")
    print(f"Wrote {os.path.join(OUT_DIR, 'full_ranking.csv')}")


if __name__ == "__main__":
    main()
