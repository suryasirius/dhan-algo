"""
Fixed-Strike ORB -- ONE trade/day, NO stop -- FULL institutional metrics suite
=================================================================
Extends nifty_fixed_strike_1trade_timesweep.py (untouched, not modified)
in two ways:

  1. Adds a TARGET-POINTS sweep on top of the existing hold-time sweep:
     targets = 5,6,7,8,9,10,11,12,13 points  x  holds = 90sec,2min,3min,
     4min,5min  =  45 combinations. Same base rules throughout: ONE
     fixed nearest-ITM CE strike per day, ONE trade per day (no
     re-entry), NO stop-loss, exit at target OR the max-hold deadline
     OR EOD, whichever comes first.

  2. Wraps every combination in the full performance-metrics stack you
     asked for: CAGR, Sharpe, Sortino, max drawdown % + duration, win
     rate, avg win/loss ratio, a t-test that mean daily return != 0,
     Value-at-Risk / Conditional VaR (95%), correlation/alpha/beta vs a
     buy-and-hold Nifty benchmark, an out-of-sample train/test split, a
     quarterly walk-forward re-selection, a volatility-regime cross-
     validation (low/med/high OR-range terciles), and a proper White's
     Reality-Check-style bootstrap that corrects for having searched 45
     parameter combinations (data-mining bias) rather than just one.

DEFAULT WINDOW CHANGED to 2 YEARS (was 6 months in the base sweep file)
-- needed for CAGR/Sharpe to mean anything and for quarterly walk-
forward folds to exist at all. Flagged here and in the run output.

NO SCIPY DEPENDENCY: t-test p-values use the normal approximation to
the t-distribution (valid once n is a few dozen or more, which every
combo here comfortably clears) so this still runs on a bare pandas +
numpy install.

STANDING RISK CAVEAT (repeated deliberately): this whole strategy
family has NO STOP-LOSS. However good Sharpe/CAGR look in-sample,
single-trade downside within a session is technically unbounded until
EOD squareoff. Every report section below still says so.

PROXY PREMIUM MODE: same convention as the rest of this project
(delta=0.6, +40pt time value, floored at 0) -- real Dhan option-premium
data is still blocked (DH-901).
"""

import datetime as dt
import math
import numpy as np
import pandas as pd

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")   # 2 years (was 6mo)
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OR_CANDLES = 5
STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)

TARGET_VALUES = [5, 6, 7, 8, 9, 10, 11, 12, 13]
HOLD_DURATIONS = [
    ("90sec", pd.Timedelta(seconds=90)),
    ("2min", pd.Timedelta(minutes=2)),
    ("3min", pd.Timedelta(minutes=3)),
    ("4min", pd.Timedelta(minutes=4)),
    ("5min", pd.Timedelta(minutes=5)),
]

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 10_000
BROKERAGE_PER_TRADE = 40.0
TRADING_DAYS_PER_YEAR = 252

BOOTSTRAP_ITERS = 2000
VOL_REGIME_LOOKBACK = 20


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
            from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
        )
        data = resp.get("data", resp)
        df = pd.DataFrame({
            "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
            "volume": data["volume"], "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def build_ce_premium_series(day_df, strike, day_open_spot):
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()

    intrinsic0 = max(day_open_spot - strike, 0.0)
    base = intrinsic0 + ASSUMED_TIME_VALUE

    p_open = np.maximum(base + (idx_open - day_open_spot) * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + (idx_close - day_open_spot) * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + (idx_high - day_open_spot) * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + (idx_low - day_open_spot) * ASSUMED_DELTA, 0.0)

    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close},
                         index=day_df.index)


def find_entry_for_day(prem_df):
    day_high = prem_df["high"].iloc[:OR_CANDLES].max()
    closes = prem_df["close"].tolist()
    for i in range(OR_CANDLES, len(prem_df)):
        if closes[i] > day_high:
            return i, day_high
    return None, day_high


def simulate_trade_fast(idx_list, closes, highs, entry_idx, target_pts, hold_delta, lot):
    """Same mechanics as the base sweep file's simulate_trade, but takes plain
    python lists (not a DataFrame) so it's fast enough to call 45x per day."""
    entry_premium = closes[entry_idx]
    target = entry_premium + target_pts
    deadline = idx_list[entry_idx] + hold_delta
    n = len(idx_list)

    exit_i, exit_premium, reason = n - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, n):
        t = idx_list[i]
        if highs[i] >= target:
            exit_i, exit_premium, reason = i, target, "TARGET"
            break
        if t >= deadline:
            exit_i, exit_premium, reason = i, closes[i], "TIME"
            break
        if t.time() >= EOD_SQUAREOFF:
            exit_i, exit_premium, reason = i, closes[i], "EOD"
            break

    points = exit_premium - entry_premium
    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return points, rupee_pnl, reason


# ----------------------------- metrics helpers -----------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def t_test_vs_zero(returns):
    """One-sample t-test that mean(returns) != 0. Uses the normal
    approximation to the t-distribution p-value (no scipy dependency);
    valid once n is a few dozen or more, comfortably true here."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0:
        return dict(t_stat=float("nan"), p_value=float("nan"), n=n)
    t_stat = r.mean() / (r.std(ddof=1) / math.sqrt(n))
    p_value = 2 * (1 - norm_cdf(abs(t_stat)))
    return dict(t_stat=t_stat, p_value=p_value, n=n)


def sharpe_ratio(returns):
    r = np.asarray(returns, dtype=float)
    if r.std(ddof=1) == 0:
        return float("nan")
    return (r.mean() / r.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns):
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return float("nan") if len(downside) == 0 else float("inf")
    return (r.mean() / downside.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown_pct_and_duration(equity, dates):
    """Duration convention: counts days strictly BELOW the running peak, up
    to and including the last such day before a NEW peak is set (the day a
    new peak is reached is not itself counted as 'still underwater')."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity / peak - 1.0) * 100
    max_dd_pct = dd_pct.min()

    max_dur = 0
    peak_date_idx = 0
    cur_peak = equity[0]
    for i in range(len(equity)):
        if equity[i] >= cur_peak:
            cur_peak = equity[i]
            peak_date_idx = i
        else:
            dur = (dates[i] - dates[peak_date_idx]).days
            max_dur = max(max_dur, dur)
    return max_dd_pct, max_dur


def var_cvar(returns, conf=0.95):
    r = np.asarray(returns, dtype=float)
    cutoff = np.percentile(r, (1 - conf) * 100)
    var = -cutoff
    tail = r[r <= cutoff]
    cvar = -tail.mean() if len(tail) else float("nan")
    return var, cvar


def cagr(equity_start, equity_end, calendar_days):
    if calendar_days <= 0 or equity_start <= 0:
        return float("nan")
    years = calendar_days / 365.25
    if years <= 0:
        return float("nan")
    return ((equity_end / equity_start) ** (1 / years) - 1) * 100


def compute_full_metrics(daily_pnl, dates, benchmark_returns=None):
    """daily_pnl: rupee P&L per FULL trading day (0 on no-entry days).
    dates: matching list of date objects. benchmark_returns: optional
    same-length array of the day's own index % return, for correlation/
    alpha-beta/information-ratio."""
    equity = STARTING_CAPITAL + np.cumsum(daily_pnl)
    daily_returns = np.asarray(daily_pnl, dtype=float) / STARTING_CAPITAL

    total_return_pct = (equity[-1] / STARTING_CAPITAL - 1) * 100
    calendar_days = (dates[-1] - dates[0]).days
    c = cagr(STARTING_CAPITAL, equity[-1], calendar_days)
    sharpe = sharpe_ratio(daily_returns)
    sortino = sortino_ratio(daily_returns)
    max_dd_pct, max_dd_dur = max_drawdown_pct_and_duration(equity, dates)
    var95, cvar95 = var_cvar(daily_returns)
    ttest = t_test_vs_zero(daily_returns)

    trade_days_pnl = daily_pnl[daily_pnl != 0]
    win_rate = (trade_days_pnl > 0).mean() * 100 if len(trade_days_pnl) else float("nan")
    wins = trade_days_pnl[trade_days_pnl > 0]
    losses = trade_days_pnl[trade_days_pnl <= 0]
    avg_win_loss_ratio = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan")

    out = dict(
        n_trades=len(trade_days_pnl), total_return_pct=total_return_pct, cagr_pct=c,
        sharpe=sharpe, sortino=sortino, max_dd_pct=max_dd_pct, max_dd_dur_days=max_dd_dur,
        win_rate_pct=win_rate, avg_win_loss_ratio=avg_win_loss_ratio,
        var95_pct=var95 * 100, cvar95_pct=cvar95 * 100,
        t_stat=ttest["t_stat"], p_value=ttest["p_value"],
        final_equity=equity[-1],
    )

    if benchmark_returns is not None:
        b = np.asarray(benchmark_returns, dtype=float)
        if b.std(ddof=1) > 0:
            corr = np.corrcoef(daily_returns, b)[0, 1]
            beta, alpha = np.polyfit(b, daily_returns, 1)
            excess = daily_returns - b
            info_ratio = (excess.mean() / excess.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR) if excess.std(ddof=1) > 0 else float("nan")
            out.update(corr_vs_benchmark=corr, alpha_daily=alpha, alpha_annualized_pct=alpha * TRADING_DAYS_PER_YEAR * 100,
                        beta_vs_benchmark=beta, information_ratio=info_ratio)
    return out


def reality_check_bootstrap(pnl_matrix, n_boot=BOOTSTRAP_ITERS, seed=42):
    """White's Reality-Check-style correction for having searched many (here
    45) parameter combinations. pnl_matrix: (n_days x n_combos) array of
    daily P&L (0 on no-entry days), same day-index shared across combos.

    Demeans each combo's daily-return series to enforce the null of 'no
    genuine average edge', then bootstrap-resamples days (same resampled
    day-index applied to every combo at once, preserving cross-combo
    correlation) to build the null distribution of the BEST-of-N Sharpe.
    Returns (observed_best_sharpe, null_distribution, p_value).
    """
    rng = np.random.default_rng(seed)
    returns_matrix = pnl_matrix / STARTING_CAPITAL   # (n_days, n_combos)
    n_days, n_combos = returns_matrix.shape

    observed_sharpes = np.array([sharpe_ratio(returns_matrix[:, k]) for k in range(n_combos)])
    observed_best_sharpe = np.nanmax(observed_sharpes)

    demeaned = returns_matrix - returns_matrix.mean(axis=0, keepdims=True)

    null_best = np.empty(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_days, size=n_days)
        resampled = demeaned[sample_idx, :]
        means = resampled.mean(axis=0)
        stds = resampled.std(axis=0, ddof=1)
        stds[stds == 0] = np.nan
        sharpes = (means / stds) * math.sqrt(TRADING_DAYS_PER_YEAR)
        null_best[b] = np.nanmax(sharpes)

    p_value = (null_best >= observed_best_sharpe).mean()
    return observed_best_sharpe, null_best, p_value


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    rng_idx = pd.date_range("2026-01-06 09:15", periods=12, freq="1min")
    close = [100, 100, 100, 100, 100, 100, 103, 106, 108, 110, 111, 112]
    idx_list = list(rng_idx)
    highs = [c + 0.5 for c in close]

    pts, pnl, reason = simulate_trade_fast(idx_list, close, highs, entry_idx=5, target_pts=5.0,
                                            hold_delta=pd.Timedelta(minutes=5), lot=75)
    assert reason == "TARGET" and abs(pts - 5.0) < 0.01
    lot = 75
    assert abs(pnl - (5.0 * lot - BROKERAGE_PER_TRADE)) < 0.01
    print("  simulate_trade_fast target-hit check ok")

    close2 = [100]*6 + [100.2, 100.4, 100.6, 100.8, 101.0, 101.2]
    highs2 = [c + 0.5 for c in close2]
    pts2, pnl2, reason2 = simulate_trade_fast(idx_list, close2, highs2, entry_idx=5, target_pts=5.0,
                                               hold_delta=pd.Timedelta(minutes=3), lot=75)
    assert reason2 == "TIME"
    print("  simulate_trade_fast time-exit check ok")

    # -------- metrics helper sanity checks --------
    # Sharpe of a perfectly steady positive-return series should be very large (low vol)
    steady = np.full(100, 0.001)
    steady[0] += 1e-9  # avoid exact-zero std
    sh = sharpe_ratio(steady)
    assert sh > 0 and not math.isnan(sh)
    print(f"  sharpe_ratio steady-positive-series check ok (sharpe={sh:.1f})")

    # A clearly positive-mean series must produce a tiny p-value (rejects H0: mean=0)
    strong_pos = np.random.default_rng(1).normal(0.01, 0.005, 300)
    tt = t_test_vs_zero(strong_pos)
    assert tt["p_value"] < 0.001, f"expected tiny p-value for strong positive mean, got {tt['p_value']}"
    # A zero-mean symmetric series should NOT reject H0 (large p-value)
    zero_mean = np.random.default_rng(2).normal(0.0, 0.01, 300)
    tt0 = t_test_vs_zero(zero_mean)
    assert tt0["p_value"] > 0.05, f"expected large p-value for zero-mean series, got {tt0['p_value']}"
    print("  t_test_vs_zero strong-signal vs null-signal check ok")

    # VaR/CVaR on a known small dataset
    known = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) / 100.0
    v, cv = var_cvar(known, conf=0.95)
    # 5th percentile of 20 sorted points -> near the bottom -> var should be positive (a loss)
    assert v > 0, f"expected positive VaR (a loss), got {v}"
    assert cv >= v - 1e-9, "CVaR should be at least as large as VaR (tail mean >= tail cutoff)"
    print("  var_cvar sanity check ok")

    # Max drawdown + duration on a known synthetic equity curve
    eq = [100, 110, 105, 95, 90, 100, 115, 120]
    dts = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(len(eq))]
    mdd_pct, mdd_dur = max_drawdown_pct_and_duration(eq, dts)
    # peak 110 (day1) -> trough 90 (day4) -> dd% = 90/110-1 = -18.18%
    # days spent strictly BELOW the day1 peak: day2..day5 (day6 sets a new peak of 115) -> 4 days
    # measured as (last-underwater-day - peak-day).days = (day5 - day1).days = 4
    assert abs(mdd_pct - (90/110 - 1)*100) < 0.01, f"expected ~-18.18%, got {mdd_pct}"
    assert mdd_dur == 4, f"expected 4-day drawdown duration, got {mdd_dur}"
    print("  max_drawdown_pct_and_duration known-curve check ok")

    # CAGR sanity: doubling equity over exactly 1 year -> CAGR = 100%
    c1 = cagr(10000, 20000, 365)
    assert abs(c1 - 100.0) < 1.0, f"expected ~100% CAGR, got {c1}"
    print("  cagr known-case check ok")

    # Correlation/alpha/beta sanity: perfectly correlated series (strategy = 2x benchmark) -> corr=1, beta=2
    bench = np.random.default_rng(3).normal(0.0005, 0.01, 200)
    strat_perfect = 2.0 * bench
    metrics = compute_full_metrics(
        strat_perfect * STARTING_CAPITAL,
        [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(200)],
        benchmark_returns=bench,
    )
    assert abs(metrics["corr_vs_benchmark"] - 1.0) < 1e-6, f"expected corr=1, got {metrics['corr_vs_benchmark']}"
    assert abs(metrics["beta_vs_benchmark"] - 2.0) < 1e-6, f"expected beta=2, got {metrics['beta_vs_benchmark']}"
    print("  compute_full_metrics correlation/alpha/beta known-case check ok")

    # Reality-check bootstrap: with genuinely random (no-edge) combos, the OBSERVED best
    # Sharpe should land comfortably inside its own null distribution (p-value not tiny).
    rng2 = np.random.default_rng(5)
    noise_pnl_matrix = rng2.normal(0, 500, size=(250, 45))  # 45 pure-noise "combos"
    obs_best, null_dist, pval = reality_check_bootstrap(noise_pnl_matrix, n_boot=300, seed=7)
    assert 0.0 <= pval <= 1.0
    assert pval > 0.03, f"pure noise should rarely look 'significant' at p<0.03, got p={pval}"
    print(f"  reality_check_bootstrap pure-noise-not-falsely-significant check ok (p={pval:.3f})")

    print("All self-checks passed.\n")


def classify_vol_regime(or_ranges):
    """Tercile split of opening-range size relative to a trailing 20-day
    average -> Low/Med/High volatility regime label per day."""
    s = pd.Series(or_ranges)
    roll_avg = s.shift(1).rolling(VOL_REGIME_LOOKBACK, min_periods=5).mean()
    ratio = s / roll_avg
    labels = pd.Series(index=s.index, dtype=object)
    valid = ratio.dropna()
    if len(valid) >= 3:
        q1, q2 = valid.quantile([1/3, 2/3])
        labels[ratio <= q1] = "Low"
        labels[(ratio > q1) & (ratio <= q2)] = "Med"
        labels[ratio > q2] = "High"
    labels[ratio.isna()] = "Med"  # warmup days default to Med
    return labels.tolist()


def main():
    run_self_checks()
    print("=== Fixed-Strike ORB -- 1 trade/day, NO stop -- FULL METRICS SUITE (target x hold sweep) ===\n")
    print(f"Target sweep: {TARGET_VALUES} points   Hold sweep: {[l for l,_ in HOLD_DURATIONS]}")
    print(f"{len(TARGET_VALUES)} x {len(HOLD_DURATIONS)} = {len(TARGET_VALUES)*len(HOLD_DURATIONS)} combinations\n")
    print("WINDOW: 2 years (changed from the base sweep file's 6-month default -- needed for CAGR, ")
    print("Sharpe, and quarterly walk-forward folds to be meaningful).\n")
    print("RISK CAVEAT: this strategy has NO STOP-LOSS. Every metric below can look good in-sample")
    print("while a single bad trade's downside remains technically unbounded until EOD squareoff.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    all_days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(all_days)} trading day(s): {all_days[0]} to {all_days[-1]}\n")

    combos = [(tgt, label, delta) for tgt in TARGET_VALUES for label, delta in HOLD_DURATIONS]
    combo_keys = [f"t{tgt}_{label}" for tgt, label, _ in combos]

    pnl_by_day = {k: [] for k in combo_keys}
    bench_return_by_day = []
    or_range_by_day = []
    day_has_entry = []

    for d in all_days:
        day_df = df1m[df1m.index.date == d]
        if len(day_df) < OR_CANDLES + 1:
            for k in combo_keys:
                pnl_by_day[k].append(0.0)
            bench_return_by_day.append(0.0)
            or_range_by_day.append(np.nan)
            day_has_entry.append(False)
            continue

        day_open_spot = day_df["open"].iloc[0]
        day_close_spot = day_df["close"].iloc[-1]
        bench_ret = (day_close_spot - day_open_spot) / day_open_spot
        bench_return_by_day.append(bench_ret)

        or_range = (day_df["high"].iloc[:OR_CANDLES].max() - day_df["low"].iloc[:OR_CANDLES].min())
        or_range_by_day.append(or_range)

        strike = itm_call_strike(day_open_spot)
        prem_df = build_ce_premium_series(day_df, strike, day_open_spot)
        entry_idx, _ = find_entry_for_day(prem_df)

        if entry_idx is None:
            for k in combo_keys:
                pnl_by_day[k].append(0.0)
            day_has_entry.append(False)
            continue

        day_has_entry.append(True)
        lot = lot_size_for_date(d)
        idx_list = list(prem_df.index)
        closes = prem_df["close"].tolist()
        highs = prem_df["high"].tolist()

        for (tgt, label, delta), key in zip(combos, combo_keys):
            _, pnl, _ = simulate_trade_fast(idx_list, closes, highs, entry_idx, tgt, delta, lot)
            pnl_by_day[key].append(pnl)

    print(f"Days with a qualifying entry: {sum(day_has_entry)} / {len(all_days)}\n")

    pnl_matrix = np.array([pnl_by_day[k] for k in combo_keys]).T  # (n_days, n_combos)
    bench_returns = np.array(bench_return_by_day)

    # ---------------- 1) full grid: core + risk metrics for all 45 combos ----------------
    grid_rows = []
    for j, key in enumerate(combo_keys):
        m = compute_full_metrics(pnl_matrix[:, j], all_days, benchmark_returns=bench_returns)
        m["combo"] = key
        grid_rows.append(m)
    grid_df = pd.DataFrame(grid_rows).set_index("combo")
    grid_df.to_csv("nifty_fixed_strike_1trade_full_metrics_grid.csv")

    print("--- Full 45-combo grid (core + risk metrics) ---")
    cols = ["n_trades", "total_return_pct", "cagr_pct", "sharpe", "sortino", "max_dd_pct",
            "max_dd_dur_days", "win_rate_pct", "avg_win_loss_ratio", "var95_pct", "cvar95_pct",
            "p_value", "final_equity"]
    with pd.option_context("display.width", 220, "display.max_rows", 60):
        print(grid_df[cols].round(3).to_string())
    print()

    best_key = grid_df["sharpe"].idxmax()
    print(f"Best combo by Sharpe: {best_key}\n")
    best_idx = combo_keys.index(best_key)
    best_pnl = pnl_matrix[:, best_idx]

    # ---------------- 2) deep-dive report for the Sharpe-selected best combo ----------------
    print(f"=== Deep-dive: {best_key} ===\n")
    m = compute_full_metrics(best_pnl, all_days, benchmark_returns=bench_returns)
    print("--- Returns & risk ---")
    print(f"Total return: {m['total_return_pct']:.2f}%   CAGR: {m['cagr_pct']:.2f}%   "
          f"Sharpe: {m['sharpe']:.2f}   Sortino: {m['sortino']:.2f}")
    print(f"Max drawdown: {m['max_dd_pct']:.2f}%   Max DD duration: {m['max_dd_dur_days']} days")
    print(f"Win rate: {m['win_rate_pct']:.1f}%   Avg win/loss ratio: {m['avg_win_loss_ratio']:.2f}")
    print(f"VaR(95%) daily: {m['var95_pct']:.2f}%   CVaR(95%) daily: {m['cvar95_pct']:.2f}%\n")

    print("--- Statistical significance ---")
    print(f"t-test (mean daily return != 0): t={m['t_stat']:.2f}, p-value={m['p_value']:.4f}")
    if m['p_value'] < 0.05:
        print("  -> statistically distinguishable from zero at the 5% level (on its own, BEFORE")
        print("     correcting for the 45-combo search -- see the reality-check p-value below).")
    else:
        print("  -> NOT statistically distinguishable from zero at the 5% level.")
    print(f"Correlation vs Nifty buy-and-hold: {m['corr_vs_benchmark']:.3f}   "
          f"Beta: {m['beta_vs_benchmark']:.3f}   Annualized alpha: {m['alpha_annualized_pct']:.2f}%")
    print(f"Information ratio vs benchmark: {m['information_ratio']:.2f}\n")

    print("--- Data-mining correction: White's-Reality-Check-style bootstrap over all 45 combos ---")
    obs_best_sharpe, null_dist, p_reality = reality_check_bootstrap(pnl_matrix)
    print(f"Observed best-of-45 Sharpe: {obs_best_sharpe:.2f}")
    print(f"Bootstrap null (no true edge) best-of-45 Sharpe -> 50th pct: {np.nanpercentile(null_dist,50):.2f}, "
          f"95th pct: {np.nanpercentile(null_dist,95):.2f}")
    print(f"Reality-check p-value: {p_reality:.4f}")
    if p_reality < 0.05:
        print("  -> the best combo's edge looks unlikely to be pure luck from searching 45 combos.")
    else:
        print("  -> CAUTION: a result this good could plausibly arise by chance alone from searching")
        print("     45 combos, even if none of them has a genuine edge. Treat the 'best' combo skeptically.")
    print()

    # ---------------- 3) out-of-sample train/test split (chronological 70/30) ----------------
    print("--- Robustness: out-of-sample train/test split (chronological 70/30) ---")
    n = len(all_days)
    split = int(n * 0.7)
    train_sharpes = np.array([sharpe_ratio(pnl_matrix[:split, j] / STARTING_CAPITAL) for j in range(len(combo_keys))])
    train_best_j = int(np.nanargmax(train_sharpes))
    train_best_key = combo_keys[train_best_j]
    test_metrics = compute_full_metrics(pnl_matrix[split:, train_best_j], all_days[split:])
    print(f"In-sample (first {split} days) best combo: {train_best_key}  (train Sharpe={train_sharpes[train_best_j]:.2f})")
    print(f"Applied out-of-sample (last {n-split} days): Sharpe={test_metrics['sharpe']:.2f}  "
          f"total_return={test_metrics['total_return_pct']:.2f}%  win%={test_metrics['win_rate_pct']:.1f}")
    if test_metrics['sharpe'] < train_sharpes[train_best_j] * 0.3:
        print("  -> WARNING: out-of-sample Sharpe collapsed relative to in-sample -- signs of overfitting.")
    print()

    # ---------------- 4) quarterly walk-forward (expanding window) ----------------
    print("--- Robustness: quarterly walk-forward (expanding-window re-selection) ---")
    dates_arr = pd.to_datetime(all_days)
    quarters = dates_arr.to_period("Q")
    unique_quarters = sorted(set(quarters))
    wf_pnl = []
    wf_dates = []
    wf_choices = []
    for qi in range(1, len(unique_quarters)):
        train_mask = quarters < unique_quarters[qi]
        test_mask = quarters == unique_quarters[qi]
        if train_mask.sum() < 20 or test_mask.sum() == 0:
            continue
        train_sh = np.array([sharpe_ratio(pnl_matrix[train_mask, j] / STARTING_CAPITAL) for j in range(len(combo_keys))])
        if np.all(np.isnan(train_sh)):
            continue
        chosen_j = int(np.nanargmax(train_sh))
        wf_choices.append((str(unique_quarters[qi]), combo_keys[chosen_j]))
        wf_pnl.extend(pnl_matrix[test_mask, chosen_j].tolist())
        wf_dates.extend([d for d, keep in zip(all_days, test_mask) if keep])

    if len(wf_pnl) > 10:
        wf_metrics = compute_full_metrics(np.array(wf_pnl), wf_dates)
        print(f"Quarters walked forward: {len(wf_choices)}")
        for q, ch in wf_choices:
            print(f"  {q}: selected {ch}")
        print(f"Walk-forward chained result: total_return={wf_metrics['total_return_pct']:.2f}%  "
              f"Sharpe={wf_metrics['sharpe']:.2f}  max_DD={wf_metrics['max_dd_pct']:.2f}%")
    else:
        print("Not enough quarters of data for a walk-forward chain.")
    print()

    # ---------------- 5) volatility-regime cross-validation (best combo only) ----------------
    print(f"--- Cross-validation across volatility regimes ({best_key}) ---")
    regime_labels = classify_vol_regime(or_range_by_day)
    reg_df = pd.DataFrame({"regime": regime_labels, "pnl": best_pnl})
    for reg in ["Low", "Med", "High"]:
        g = reg_df[reg_df["regime"] == reg]
        trades = g[g["pnl"] != 0]
        if len(trades) == 0:
            print(f"  {reg} volatility: no trades")
            continue
        win_pct = (trades["pnl"] > 0).mean() * 100
        print(f"  {reg} volatility: n={len(trades)}  win%={win_pct:.1f}  net_Rs={trades['pnl'].sum():,.2f}")
    print()

    print(f"Full grid saved: nifty_fixed_strike_1trade_full_metrics_grid.csv ({len(grid_df)} rows)")


if __name__ == "__main__":
    main()
