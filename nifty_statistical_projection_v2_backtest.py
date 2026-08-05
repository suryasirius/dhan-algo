"""
NIFTY STATISTICAL PROJECTION ENGINE (NSPE) v2 -- REFINED -- 2-YEAR BACKTEST
(+ separate 6-month breakdown from the SAME 2-year data, one fetch)
=================================================================
Refines nifty_statistical_projection_backtest.py (v1 MVP: 4-day window
only, fixed-point stop/trail, no target) per your v2.0 spec. What's
NEWLY added this version, and what's still scoped out -- stated plainly
so nothing is silently skipped:

  ADDED -- Multi-window blend (Step 1/7): 4/10/20/50-day windows, each
  blended with window-weights 40/30/20/10 exactly as your table states.
  Within each window, days are recency-weighted by a linear decay
  (yesterday heaviest) -- your spec only gave explicit day-weights for
  the 4-day window, so the same linear-decay principle is applied to the
  larger windows too (documented assumption, not hidden).

  ADDED -- Confidence Engine (Step 9), operationalized as: how many of
  the 4 windows INDEPENDENTLY agree price has broken their own mean-high
  (or mean-low for PE). With exactly 4 discrete windows, confidence can
  only be 0/25/50/75/100% -- your spec's "80%" isn't reachable on this
  scale, so the threshold used is >=75% (at least 3 of 4 windows agree),
  the honest nearest equivalent.

  ADDED -- Breakout Quality Score (Step 10): in this version it equals
  the Confidence score above. Your spec lists 6 sub-factors (multi-
  window agreement, gap alignment, regime, volatility, time-of-day
  reliability, projection confidence) -- but gap and volatility are
  already hard PASS/FAIL gates applied before scoring (an entry that
  fails either never reaches the score at all), and "regime" /
  "time-of-day reliability" require the full regime-classification and
  per-5-min-slot sub-systems that are scoped out below. So the only
  genuinely independent, gradable factor left is window agreement --
  using it directly as the score is the honest version of this, not an
  invented number dressed up as a 6-factor blend.

  ADDED -- Dynamic stop-loss (Step 12) and a real statistical TARGET
  (Step 13, v1 had no target at all): stop and trail are now sized as a
  FRACTION of the blended average daily range (ATR-like, not fixed
  points), and the target is the 75th percentile of the last 50 days'
  daily highs (CE) / 25th percentile of daily lows (PE) -- your "Target
  2" from the 3-stage list. Target 1 (mean high) and Target 3 (95th
  percentile) are not separately staged in this version -- a genuine
  3-target progressive exit is a real follow-up, not part of this pass.

  STILL SCOPED OUT (same reasoning as v1, now explicit for v2 too):
  regime classification (trend/range/reversal/expansion/compression day),
  per-5-minute-candle statistical profiles, projection-error learning /
  failure-analysis database, walk-forward validation loop, continuous
  re-calibration. These are real, valuable parts of your spec but are
  each their own sub-project -- listed here so they're not forgotten,
  not because they don't matter.

Does NOT touch nifty_statistical_projection_backtest.py (v1, already
delivered) or any other locked file. Reuses top3's premium-building,
lot-size, and EOD-squareoff conventions; base's fetch/atm_strike.

2-year backtest window, Rs25,000 capital, plus a SEPARATE 6-month
breakdown sliced from the same 2-year trade set (one data fetch, two
reported views, per your request).

Run:
    cd "C:\\dhan algo"
    python nifty_statistical_projection_v2_backtest.py --smoke-test
    python nifty_statistical_projection_v2_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as base
import nifty_statistical_projection_backtest as v1
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

BACKTEST_DAYS = 730  # 2 years
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
RECENT_6MO_DAYS = 183

LOOKBACK_WINDOWS = [4, 10, 20, 50]
WINDOW_WEIGHTS = {4: 0.4, 10: 0.3, 20: 0.2, 50: 0.1}
CONFIDENCE_THRESHOLD = 0.75   # >=3 of 4 windows must agree
GAP_THRESHOLD_FRAC = 0.3
VOL_THRESHOLD_FRAC = 0.3

STOP_FRAC_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]   # fraction of blended_avg_range
TRAIL_FRAC_GRID = [0.0, 0.05, 0.08, 0.10, 0.15]
DEFAULT_STOP_FRAC, DEFAULT_TRAIL_FRAC = 0.20, 0.08
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000
TRADES_CSV = "nifty_statistical_projection_v2_trades.csv"


# =============================================================================
# Multi-window projection
# =============================================================================
def linear_recency_weights(n):
    raw = list(range(n, 0, -1))
    s = sum(raw)
    return [r / s for r in raw]


def window_stats(daily_ohlc, target_day, window):
    prior = daily_ohlc[daily_ohlc.index.date < target_day]
    if len(prior) < window:
        return None
    recent = prior.iloc[-window:][::-1]
    w = linear_recency_weights(window)
    highs, lows = recent["high"].tolist(), recent["low"].tolist()
    ranges = (recent["high"] - recent["low"]).tolist()
    return {
        "mean_high": sum(v * ww for v, ww in zip(highs, w)),
        "mean_low": sum(v * ww for v, ww in zip(lows, w)),
        "avg_range": sum(v * ww for v, ww in zip(ranges, w)),
    }


def compute_multiwindow_projection(daily_ohlc, target_day):
    stats = {}
    for wlen in LOOKBACK_WINDOWS:
        s = window_stats(daily_ohlc, target_day, wlen)
        if s is None:
            return None
        stats[wlen] = s
    blended_mean_high = sum(stats[w]["mean_high"] * WINDOW_WEIGHTS[w] for w in LOOKBACK_WINDOWS)
    blended_mean_low = sum(stats[w]["mean_low"] * WINDOW_WEIGHTS[w] for w in LOOKBACK_WINDOWS)
    blended_avg_range = sum(stats[w]["avg_range"] * WINDOW_WEIGHTS[w] for w in LOOKBACK_WINDOWS)
    prior = daily_ohlc[daily_ohlc.index.date < target_day]
    pool = prior.iloc[-50:]
    return {
        "blended_mean_high": blended_mean_high, "blended_mean_low": blended_mean_low,
        "blended_avg_range": blended_avg_range, "prev_close": prior["close"].iloc[-1],
        "target_high_75": pool["high"].quantile(0.75), "target_low_25": pool["low"].quantile(0.25),
        "window_mean_highs": {w: stats[w]["mean_high"] for w in LOOKBACK_WINDOWS},
        "window_mean_lows": {w: stats[w]["mean_low"] for w in LOOKBACK_WINDOWS},
    }


def find_entry_for_day(day_df_1min, proj):
    if proj is None or len(day_df_1min) < 2:
        return None, None, None
    day_open = day_df_1min["open"].iloc[0]
    gap = day_open - proj["prev_close"]
    gap_down_strong = gap < -GAP_THRESHOLD_FRAC * proj["blended_avg_range"]
    gap_up_strong = gap > GAP_THRESHOLD_FRAC * proj["blended_avg_range"]
    vol_floor = VOL_THRESHOLD_FRAC * proj["blended_avg_range"]

    closes = day_df_1min["close"].to_numpy()
    highs = day_df_1min["high"].to_numpy()
    lows = day_df_1min["low"].to_numpy()
    running_high, running_low = highs[0], lows[0]

    for i in range(len(day_df_1min)):
        running_high = max(running_high, highs[i])
        running_low = min(running_low, lows[i])
        vol_ok = (running_high - running_low) >= vol_floor
        price = closes[i]

        if (not gap_down_strong) and vol_ok and price > proj["blended_mean_high"]:
            agree = sum(1 for w in LOOKBACK_WINDOWS if price > proj["window_mean_highs"][w])
            conf = agree / len(LOOKBACK_WINDOWS)
            if conf >= CONFIDENCE_THRESHOLD:
                return i, "CE", conf
        if (not gap_up_strong) and vol_ok and price < proj["blended_mean_low"]:
            agree = sum(1 for w in LOOKBACK_WINDOWS if price < proj["window_mean_lows"][w])
            conf = agree / len(LOOKBACK_WINDOWS)
            if conf >= CONFIDENCE_THRESHOLD:
                return i, "PE", conf
    return None, None, None


def index_to_premium(index_price, strike, day_open_spot, side):
    intrinsic0 = max(day_open_spot - strike, 0.0) if side == "CE" else max(strike - day_open_spot, 0.0)
    base_val = intrinsic0 + top3.ASSUMED_TIME_VALUE
    move = (index_price - day_open_spot) if side == "CE" else (day_open_spot - index_price)
    return max(base_val + move * top3.ASSUMED_DELTA, 0.0)


def step_with_target(pos, premium_df, trail_pts):
    """Same conservative ordering as top3.step_long_premium_position (stop
    checked on this bar's low before anything favorable), with a TARGET
    check inserted after the EOD check and before the trail update."""
    n = len(premium_df)
    for j in range(pos["last_checked_idx"] + 1, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= pos["stop_level"]:
            reason = "TRAIL_SL" if pos["trailing_active"] else "INITIAL_SL"
            return {"exit_price": pos["stop_level"], "reason": reason,
                    "exit_time": premium_df.index[j], "exit_idx": j}
        if t >= top3.EOD_SQUAREOFF:
            return {"exit_price": bar["close"], "reason": "EOD",
                    "exit_time": premium_df.index[j], "exit_idx": j}
        if bar["high"] >= pos["target_level"]:
            return {"exit_price": pos["target_level"], "reason": "TARGET",
                    "exit_time": premium_df.index[j], "exit_idx": j}
        if bar["high"] > pos["peak"]:
            pos["peak"] = bar["high"]
        if pos["peak"] > pos["entry_price"]:
            pos["trailing_active"] = True
        if pos["trailing_active"]:
            pos["stop_level"] = max(pos["stop_level"], pos["peak"] - trail_pts)
        pos["last_checked_idx"] = j
    return None


def run_standalone_for_day(day_df_1min, daily_ohlc):
    n = len(day_df_1min)
    if n < 5:
        return None
    day = day_df_1min.index[0].date()
    proj = compute_multiwindow_projection(daily_ohlc, day)
    if proj is None:
        return None
    sig_idx, side, conf = find_entry_for_day(day_df_1min, proj)
    if sig_idx is None:
        return None
    exec_idx = sig_idx + 1
    if exec_idx >= n:
        return None
    day_open_spot = day_df_1min["open"].iloc[0]
    strike = base.atm_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df_1min, strike, day_open_spot, side=side)
    lot = top3.lot_size_for_date(day)
    entry_price = premium_df["open"].iloc[exec_idx]
    target_idx_level = proj["target_high_75"] if side == "CE" else proj["target_low_25"]
    target_premium = index_to_premium(target_idx_level, strike, day_open_spot, side)
    return {
        "day": day, "premium_df": premium_df, "exec_idx": exec_idx, "side": side, "strike": strike,
        "lot": lot, "entry_price": entry_price, "confidence": conf,
        "blended_avg_range": proj["blended_avg_range"], "target_premium": max(target_premium, entry_price + 0.5),
    }


def simulate_combo(contexts, stop_frac, trail_frac):
    trades = []
    for ctx in contexts:
        premium_df = ctx["premium_df"]
        exec_idx = ctx["exec_idx"]
        entry_price = ctx["entry_price"]
        stop_pts = stop_frac * ctx["blended_avg_range"] * top3.ASSUMED_DELTA
        trail_pts = (trail_frac * ctx["blended_avg_range"] * top3.ASSUMED_DELTA) if trail_frac > 0 else 1e9
        pos = {"entry_price": entry_price, "stop_level": entry_price - stop_pts, "peak": entry_price,
               "trailing_active": False, "last_checked_idx": exec_idx - 1, "target_level": ctx["target_premium"]}
        exit_info = step_with_target(pos, premium_df, trail_pts)
        if exit_info is None:
            exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": premium_df.index[-1], "exit_idx": len(premium_df) - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        rupee_pnl = pnl_pts * ctx["lot"] - BROKERAGE_PER_TRADE
        trades.append({
            "date": ctx["day"], "side": ctx["side"], "strike": ctx["strike"], "confidence": ctx["confidence"],
            "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
            "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
            "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl,
        })
    return pd.DataFrame(trades)


def compute_pf_sqn(trades_df):
    if trades_df is None or trades_df.empty:
        return 0.0, 0.0, 0
    pnl = trades_df["rupee_pnl"]
    n = len(pnl)
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    std = pnl.std(ddof=1) if n > 1 else 0.0
    sqn = (pnl.mean() / std * math.sqrt(n)) if std > 0 else 0.0
    return pf, sqn, n


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top3.run_self_checks()
    print("  top3 primitives re-verified ok")

    w = linear_recency_weights(4)
    assert abs(sum(w) - 1.0) < 1e-9 and w[0] > w[1] > w[2] > w[3]
    print("  linear_recency_weights: sums to 1, most-recent-day heaviest ok")

    idx = pd.date_range("2024-01-01", periods=55, freq="B")
    highs = [25000 + i * 2 for i in range(55)]
    lows = [24950 + i * 2 for i in range(55)]
    closes = [24980 + i * 2 for i in range(55)]
    opens = [24960 + i * 2 for i in range(55)]
    daily = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)
    proj_short = compute_multiwindow_projection(daily, idx[10].date())  # only 10 prior days -- 50-day window fails
    assert proj_short is None
    proj_full = compute_multiwindow_projection(daily, idx[54].date())  # 54 prior days -- all windows available
    assert proj_full is not None
    assert proj_full["blended_mean_high"] < highs[-1] and proj_full["blended_mean_high"] > highs[0]
    print("  compute_multiwindow_projection: None with insufficient history, sane blended value once full ok")

    # --- confidence gate: on a monotonically rising series, price above the
    # blended mean high should agree with ALL 4 windows (since larger, older
    # windows have LOWER mean highs on a rising series) -- confidence=100% ---
    idx2 = pd.date_range("2025-01-10 09:15", periods=10, freq="1min")
    day_open = proj_full["prev_close"]
    breakout_price = proj_full["blended_mean_high"] + 50
    closes2 = [day_open] * 3 + [breakout_price] * 7
    df2 = pd.DataFrame({"open": [day_open] + closes2[:-1], "high": [c + 5 for c in closes2],
                        "low": [c - 5 for c in closes2], "close": closes2}, index=idx2)
    sig_idx, side, conf = find_entry_for_day(df2, proj_full)
    assert sig_idx is not None and side == "CE" and conf == 1.0, f"expected CE, conf=1.0, got {(sig_idx, side, conf)}"
    print(f"  full-agreement breakout (rising series, all 4 windows agree): CE fires, confidence=100% ok")

    # --- index_to_premium / step_with_target: target must fire before an
    # unrelated further stop, and stop must win if both trigger the same bar
    # (conservative ordering) ---
    strike = 25000
    day_open_spot = 25000.0
    prem_ce = index_to_premium(25050.0, strike, day_open_spot, "CE")
    expected = max((25000 - 25000) + top3.ASSUMED_TIME_VALUE + (25050 - 25000) * top3.ASSUMED_DELTA, 0.0)
    assert abs(prem_ce - expected) < 1e-9
    idx3 = pd.date_range("2025-01-10 09:20", periods=5, freq="1min")
    entry_price = 50.0
    prem_df = pd.DataFrame({"open": [50, 55, 60, 65, 70], "high": [52, 57, 63, 68, 72],
                            "low": [49, 54, 59, 64, 69], "close": [51, 56, 61, 66, 71]}, index=idx3)
    pos = {"entry_price": entry_price, "stop_level": entry_price - 5, "peak": entry_price,
           "trailing_active": False, "last_checked_idx": -1, "target_level": 63.0}
    exit_info = step_with_target(pos, prem_df, trail_pts=3.0)
    assert exit_info["reason"] == "TARGET" and abs(exit_info["exit_price"] - 63.0) < 1e-9, exit_info
    print("  step_with_target: hits the 63.0 target exactly, before any stop concern ok")

    pos2 = {"entry_price": entry_price, "stop_level": entry_price - 1, "peak": entry_price,
            "trailing_active": False, "last_checked_idx": -1, "target_level": 200.0}  # unreachable target
    exit_info2 = step_with_target(pos2, prem_df, trail_pts=3.0)
    assert exit_info2["reason"] in ("INITIAL_SL", "TRAIL_SL"), exit_info2
    print("  step_with_target: falls back to stop-loss correctly when target is unreachable ok")

    print("All self-checks passed.\n")


# =============================================================================
# Full pipeline
# =============================================================================
def precompute_day_contexts(df1m):
    daily_ohlc = v1.compute_daily_ohlc(df1m)
    days_covered = sorted(set(df1m.index.date))
    contexts = []
    for day in days_covered:
        day_df = df1m[df1m.index.date == day]
        ctx = run_standalone_for_day(day_df, daily_ohlc)
        if ctx is not None:
            contexts.append(ctx)
    return contexts, days_covered


def report_block(trades_df, label):
    pf, sqn, n = compute_pf_sqn(trades_df)
    if n == 0:
        print(f"\n################ {label} ################\nNo trades in this window.")
        return
    net = trades_df["rupee_pnl"].sum()
    win = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"\n################ {label} ################")
    print(f"Trades: {n}   Win rate: {win:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{net:,.2f}")
    if pf > 2.5 or (not math.isnan(sqn) and sqn > 5):
        print("  *** RED FLAG: PF>2.5 or SQN>5 -- treat as needing a bug hunt before trusting. ***")
    for reason, g in trades_df.groupby("reason"):
        print(f"  {reason:<14} n={len(g):<5} ({100*len(g)/n:5.1f}%)  net=Rs{g['rupee_pnl'].sum():>10,.2f}")
    equity_curve = STARTING_CAPITAL + trades_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"  Worst point-in-time drawdown (Rs{STARTING_CAPITAL:,} capital): Rs{worst_dd:,.2f}")
    daily_pnl = build_daily_pnl(trades_df.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    print(f"  CAGR: {m['cagr_pct']}%   Sharpe: {m['sharpe_fulltime']}   Sortino: {m['sortino_fulltime']}   "
          f"Max DD: {m['max_dd_pct']}%")
    print(f"  t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'significant' if m['p_value'] < 0.05 else 'NOT significant'})")


def run_full_pipeline(df1m):
    print("\nPrecomputing entry signals once (reused across every stop/trail combo)...")
    contexts, days_covered = precompute_day_contexts(df1m)
    print(f"{len(contexts)} days with a completed entry signal (out of {len(days_covered)} total days, "
          f"first ~50 days have no signal by construction -- 50-day window warmup).")
    if not contexts:
        print("No signals at all -- nothing to sweep.")
        return

    ce_n = sum(1 for c in contexts if c["side"] == "CE")
    print(f"Direction split: CE={ce_n}  PE={len(contexts)-ce_n}")
    print(f"Entry-frequency: {len(contexts)}/{len(days_covered)} days fire ({100*len(contexts)/len(days_covered):.1f}%)")
    if len(contexts) / len(days_covered) > 0.95:
        print("  *** WARNING: firing on >95% of days. ***")

    print(f"\n################ DYNAMIC STOP/TRAIL GRID SWEEP (fractions of blended avg range) ################")
    results = []
    for stop_f in STOP_FRAC_GRID:
        for trail_f in TRAIL_FRAC_GRID:
            tdf = simulate_combo(contexts, stop_f, trail_f)
            pf, sqn, n = compute_pf_sqn(tdf)
            net = tdf["rupee_pnl"].sum() if n else 0.0
            win = 100 * (tdf["rupee_pnl"] > 0).mean() if n else 0.0
            print(f"  stop_frac={stop_f:<5} trail_frac={trail_f:<5} n={n:<5} win%={win:5.1f}  "
                  f"PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")
            results.append({"stop_frac": stop_f, "trail_frac": trail_f, "pf": pf, "sqn": sqn, "n": n, "net": net})

    res_df = pd.DataFrame(results)
    best = res_df.sort_values("sqn", ascending=False).iloc[0]
    print(f"\nBest by SQN: stop_frac={best['stop_frac']} trail_frac={best['trail_frac']}  SQN={best['sqn']:.2f}  "
          f"PF={best['pf']:.2f}  net=Rs{best['net']:,.2f}")
    at_edge = (best["stop_frac"] in (STOP_FRAC_GRID[0], STOP_FRAC_GRID[-1])) or \
              (best["trail_frac"] in (TRAIL_FRAC_GRID[0], TRAIL_FRAC_GRID[-1]))
    if at_edge:
        print("  *** GRID-EDGE WARNING: best combo sits at the edge of the tested range. ***")
    if best["pf"] > 2.5 or (not pd.isna(best["sqn"]) and best["sqn"] > 5):
        print("  *** RED FLAG: PF>2.5 or SQN>5. ***")

    default_trades = simulate_combo(contexts, DEFAULT_STOP_FRAC, DEFAULT_TRAIL_FRAC)
    default_trades = default_trades.sort_values("entry_time").reset_index(drop=True)
    default_trades.to_csv(TRADES_CSV, index=False)
    print(f"\nSaved: {TRADES_CSV} ({len(default_trades)} rows)")

    dates = pd.to_datetime(default_trades["date"])
    if len(dates.unique()) >= 2:
        mid = dates.min() + (dates.max() - dates.min()) / 2
        h1, h2 = default_trades[dates <= mid], default_trades[dates > mid]
        pf1, sqn1, n1 = compute_pf_sqn(h1)
        pf2, sqn2, n2 = compute_pf_sqn(h2)
        print(f"\nOOS split (default combo stop_frac={DEFAULT_STOP_FRAC}, trail_frac={DEFAULT_TRAIL_FRAC}):")
        print(f"  H1: n={n1} PF={pf1:.2f} SQN={sqn1:.2f} net=Rs{h1['rupee_pnl'].sum():,.2f}")
        print(f"  H2: n={n2} PF={pf2:.2f} SQN={sqn2:.2f} net=Rs{h2['rupee_pnl'].sum():,.2f}")

    report_block(default_trades, f"FULL 2-YEAR RESULT (default combo, stop_frac={DEFAULT_STOP_FRAC}, "
                                   f"trail_frac={DEFAULT_TRAIL_FRAC})")

    cutoff = dates.max() - pd.Timedelta(days=RECENT_6MO_DAYS)
    recent = default_trades[dates > cutoff]
    report_block(recent, f"RECENT 6-MONTH SLICE OF THE SAME 2-YEAR RUN (since {cutoff.date()})")

    return {"trades": default_trades, "sweep": res_df}


def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~90 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        df1m = base.fetch_index_chunked(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). Re-run on the machine with a working "
              "Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned. Try again later.")
        return
    run_full_pipeline(df1m)
    print("\nSmoke test complete -- no crashes, all self-checks passed.")


def main():
    run_self_checks()
    print("=== NIFTY STATISTICAL PROJECTION ENGINE v2 -- 2-YEAR BACKTEST ===")
    print(f"Window: {FROM_DATE} to {TO_DATE}")
    print("Fetching Nifty index 1-min data...")
    df1m = base.fetch_index_chunked(FROM_DATE, TO_DATE)
    if df1m.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return
    days_covered = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")
    run_full_pipeline(df1m)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main()
