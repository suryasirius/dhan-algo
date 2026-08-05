"""
NIFTY STATISTICAL PROJECTION ENGINE (NSPE) -- MVP v1 -- 6-MONTH BACKTEST
=================================================================
TRANSLATED SPEC (per your NSPE description, scoped down to what's
actually testable as one mechanical entry/exit rule -- see "SCOPED OUT"
below for what was deliberately left for a later phase):

  Instrument/direction: Nifty index options, ATM strike fixed at day's
  open. Both CE and PE tested, mirror logic, at most ONE trade per day
  (whichever side's condition triggers first).

  Data used: previous 4 completed trading days ONLY (your stated
  DEFAULT window -- the 10/20/50-day multi-timeframe blend is scoped
  out, see below). Recency-weighted 40/30/20/10 (yesterday heaviest),
  exactly as you specified.

  Daily statistics computed per day: High, Low, Close, daily range
  (High-Low). From the last 4 days, before today's own first bar:
    w_mean_high = 0.4*H(-1) + 0.3*H(-2) + 0.2*H(-3) + 0.1*H(-4)
    w_mean_low  = same formula on Low
    w_avg_range = same formula on (High-Low)
  These are your "Projected High / Projected Low / Average Daily Range."

  Gap classification: gap = today's open - previous close. "Strongly
  gapped" if |gap| > 0.3 * w_avg_range (your Gap Engine, reduced to the
  two categories that actually gate the entry rule per your Trade Rules
  section: gap-down blocks CE, gap-up blocks PE).

  Volatility filter: at the moment of signal, today's own range-so-far
  (running high minus running low since open) must be >= 0.3 *
  w_avg_range -- a simple stand-in for your "Normal or High volatility"
  gate (today isn't a dead/compressed session).

  Entry signal (CE): first 1-min bar whose CLOSE exceeds w_mean_high,
  AND no strong gap-down, AND the volatility filter passes. This
  collapses your "price above Median High AND Mean High AND confidence
  >80% AND bullish probability >70% AND breakout confirmed" into ONE
  mechanical trigger -- see SCOPED OUT below for why.
  Entry (PE): exact mirror, close below w_mean_low, no strong gap-up.

  Entry fill: next bar's open after the confirming close (standard
  look-ahead-safe rule, same as every other strategy here).

  Exit: STANDARD stop-loss / trailing-stop-loss grid (per this project's
  own sweep convention, applied to every strategy for comparability) --
  NOT a "statistical zone" target. See SCOPED OUT.

  One trade per day, EOD square-off 15:20, Rs25,000 capital, proxy
  premium model (build_premium_series), standard lot-size schedule,
  Rs40 brokerage.

SCOPED OUT of this MVP (explicitly, not silently dropped -- these are
real parts of your spec that are either too subjective to code as an
exact rule, or belong in the "optimizations needed" phase after this
core idea is proven or disproven):
  - Multi-timeframe 10/20/50-day blend and per-window "confidence
    engine" (agreement across windows) -- MVP uses the 4-day window
    only, your stated default.
  - Per-5-minute-candle statistical "personality" profiles (mean/median/
    bullish % per time slot) -- not used for entry/exit here.
  - "Confidence >80%" / "Bullish probability >70%" as literal numeric
    filters -- there's no way to derive a calibrated probability from a
    4-data-point sample; using a single confirmed-close breakout of the
    weighted mean level is the closest honest mechanical equivalent.
  - Statistical-zone / 95th-percentile TARGET exit -- replaced with the
    project's standard fixed stop/trail grid so this result is directly
    comparable to every other strategy already tested. A percentile-
    based target is a legitimate follow-up test, not part of this MVP.
  - Market-personality classification (trend/range/reversal day),
    projection-error learning, research engine, option-chain/OI/PCR/IV,
    market-breadth, ML layer -- all explicitly future work per your own
    "Future Enhancements" section, not part of a first backtest.

Run:
    cd "C:\\dhan algo"
    python nifty_statistical_projection_backtest.py --smoke-test
    python nifty_statistical_projection_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd
import numpy as np

import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as base  # reuse fetch_index_chunked, atm_strike, classify_session
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

BACKTEST_DAYS = 183  # 6 months, per your request
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")

LOOKBACK_DAYS = 4
WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # yesterday first, per your spec
GAP_THRESHOLD_FRAC = 0.3   # "strongly gapped" if |gap| > 0.3 * w_avg_range
VOL_THRESHOLD_FRAC = 0.3   # today's range-so-far must be >= 0.3 * w_avg_range

STOP_GRID = [4.0, 6.0, 8.0, 10.0, 12.0]   # standard project grid
TRAIL_GRID = [0.0, 2.0, 3.0, 4.0, 6.0]    # 0.0 = no trail
DEFAULT_STOP, DEFAULT_TRAIL = 8.0, 4.0
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000

TRADES_CSV = "nifty_statistical_projection_trades.csv"


# =============================================================================
# Daily stats + projection
# =============================================================================
def compute_daily_ohlc(df1m):
    daily = df1m.groupby(df1m.index.date).agg(open=("open", "first"), high=("high", "max"),
                                                low=("low", "min"), close=("close", "last"))
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def weighted_mean(values, weights=WEIGHTS):
    """values[0] = most recent (yesterday), matches WEIGHTS order."""
    return sum(v * w for v, w in zip(values, weights))


def compute_projection_for_day(daily_ohlc, target_day, lookback=LOOKBACK_DAYS, weights=WEIGHTS):
    """Uses only days STRICTLY before target_day. Returns None if fewer than
    `lookback` prior days exist."""
    prior = daily_ohlc[daily_ohlc.index.date < target_day]
    if len(prior) < lookback:
        return None
    recent = prior.iloc[-lookback:][::-1]  # most recent first, matches weights order
    highs = recent["high"].tolist()
    lows = recent["low"].tolist()
    ranges = (recent["high"] - recent["low"]).tolist()
    return {
        "w_mean_high": weighted_mean(highs, weights),
        "w_mean_low": weighted_mean(lows, weights),
        "w_avg_range": weighted_mean(ranges, weights),
        "prev_close": prior["close"].iloc[-1],
    }


# =============================================================================
# Entry detection
# =============================================================================
def find_entry_for_day(day_df_1min, projection):
    """Returns (signal_idx, side) or (None, None). CE checked and PE checked
    bar-by-bar in lockstep; whichever condition is met on the earliest bar
    wins (mutually exclusive by construction: CE needs close > w_mean_high,
    PE needs close < w_mean_low)."""
    if projection is None or len(day_df_1min) < 2:
        return None, None
    day_open = day_df_1min["open"].iloc[0]
    gap = day_open - projection["prev_close"]
    gap_down_strong = gap < -GAP_THRESHOLD_FRAC * projection["w_avg_range"]
    gap_up_strong = gap > GAP_THRESHOLD_FRAC * projection["w_avg_range"]

    closes = day_df_1min["close"].to_numpy()
    highs = day_df_1min["high"].to_numpy()
    lows = day_df_1min["low"].to_numpy()
    vol_floor = VOL_THRESHOLD_FRAC * projection["w_avg_range"]

    running_high, running_low = highs[0], lows[0]
    for i in range(len(day_df_1min)):
        running_high = max(running_high, highs[i])
        running_low = min(running_low, lows[i])
        range_so_far = running_high - running_low
        vol_ok = range_so_far >= vol_floor

        if (not gap_down_strong) and vol_ok and closes[i] > projection["w_mean_high"]:
            return i, "CE"
        if (not gap_up_strong) and vol_ok and closes[i] < projection["w_mean_low"]:
            return i, "PE"
    return None, None


def run_standalone_for_day(day_df_1min, daily_ohlc):
    n = len(day_df_1min)
    if n < 5:
        return None
    day = day_df_1min.index[0].date()
    projection = compute_projection_for_day(daily_ohlc, day)
    if projection is None:
        return None
    sig_idx, side = find_entry_for_day(day_df_1min, projection)
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
    return {
        "day": day, "premium_df": premium_df, "exec_idx": exec_idx, "side": side, "strike": strike,
        "lot": lot, "entry_price": entry_price, "w_mean_high": projection["w_mean_high"],
        "w_mean_low": projection["w_mean_low"],
    }


def simulate_combo(contexts, stop_pts, trail_pts):
    trades = []
    for ctx in contexts:
        premium_df = ctx["premium_df"]
        exec_idx = ctx["exec_idx"]
        entry_price = ctx["entry_price"]
        pos = {"entry_price": entry_price, "stop_level": entry_price - stop_pts, "peak": entry_price,
               "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, premium_df, trail_pts if trail_pts > 0 else 1e9)
        if exit_info is None:
            exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": premium_df.index[-1], "exit_idx": len(premium_df) - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        rupee_pnl = pnl_pts * ctx["lot"] - BROKERAGE_PER_TRADE
        trades.append({
            "date": ctx["day"], "side": ctx["side"], "strike": ctx["strike"],
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

    # --- weighted_mean / compute_projection_for_day ---
    assert abs(weighted_mean([100, 90, 80, 70]) - (0.4*100+0.3*90+0.2*80+0.1*70)) < 1e-9
    idx = pd.date_range("2025-01-01", periods=6, freq="B")
    daily = pd.DataFrame({
        "open": [100, 101, 102, 103, 104, 105], "high": [110, 108, 112, 106, 111, 109],
        "low": [95, 96, 97, 98, 99, 100], "close": [105, 104, 106, 102, 108, 107],
    }, index=idx)
    proj = compute_projection_for_day(daily, idx[4].date())  # 5th day, exactly 4 prior days exist
    expected_high = weighted_mean([106, 112, 108, 110])  # day4,3,2,1 highs, most-recent-first
    assert abs(proj["w_mean_high"] - expected_high) < 1e-9, f"{proj['w_mean_high']} vs {expected_high}"
    proj_short = compute_projection_for_day(daily, idx[2].date())  # only 2 prior days -- must be None
    assert proj_short is None
    print("  compute_projection_for_day: correct weighted levels, correctly None with <4 prior days ok")

    # --- find_entry_for_day: clean CE breakout ---
    idx2 = pd.date_range("2025-01-10 09:15", periods=20, freq="1min")
    closes = [24900.0] * 5 + [24950.0, 24980.0, 25010.0] + [25020.0] * 12  # closes[7]=25010 first crosses 25000
    df2 = pd.DataFrame({"open": [24900.0] + closes[:-1], "high": [c + 5 for c in closes],
                        "low": [c - 5 for c in closes], "close": closes}, index=idx2)
    proj2 = {"w_mean_high": 25000.0, "w_mean_low": 24700.0, "w_avg_range": 100.0, "prev_close": 24900.0}
    sig_idx, side = find_entry_for_day(df2, proj2)
    assert sig_idx == 7 and side == "CE", f"expected CE@7, got {(sig_idx, side)}"
    print("  find_entry_for_day: clean CE breakout detected at the right bar ok")

    # --- gap-down-strong must BLOCK a CE entry even if price crosses w_mean_high ---
    proj_gapdown = {"w_mean_high": 25000.0, "w_mean_low": 24700.0, "w_avg_range": 100.0,
                     "prev_close": 25200.0}  # open=24900 -> gap=-300, threshold=0.3*100=30 -> strongly gapped down
    sig_idx3, side3 = find_entry_for_day(df2, proj_gapdown)
    assert sig_idx3 is None, f"expected no CE entry on a strong gap-down day, got {(sig_idx3, side3)}"
    print("  gap-down-strong correctly blocks a CE entry even when price crosses w_mean_high ok")

    # --- volatility filter must BLOCK entry when today's range-so-far is too small ---
    tiny_range_closes = [24999.0] * 20  # never actually moves enough to build range, and never crosses level
    df3 = pd.DataFrame({"open": [24999.0] + tiny_range_closes[:-1], "high": [c + 0.5 for c in tiny_range_closes],
                        "low": [c - 0.5 for c in tiny_range_closes], "close": tiny_range_closes}, index=idx2)
    sig_idx4, side4 = find_entry_for_day(df3, proj2)
    assert sig_idx4 is None, "expected no entry on a dead/compressed day (never crosses level anyway, sanity check)"
    print("  a flat/dead day (no breakout, no volatility) correctly produces no signal ok")

    print("All self-checks passed.\n")


# =============================================================================
# Full pipeline
# =============================================================================
def precompute_day_contexts(df1m):
    daily_ohlc = compute_daily_ohlc(df1m)
    days_covered = sorted(set(df1m.index.date))
    contexts = []
    for day in days_covered:
        day_df = df1m[df1m.index.date == day]
        ctx = run_standalone_for_day(day_df, daily_ohlc)
        if ctx is not None:
            contexts.append(ctx)
    return contexts, days_covered


def run_full_pipeline(df1m):
    print("\nPrecomputing entry signals once (reused across every stop/trail combo)...")
    contexts, days_covered = precompute_day_contexts(df1m)
    print(f"{len(contexts)} days with a completed entry signal (out of {len(days_covered)} total days).")
    if not contexts:
        print("No signals at all -- nothing to sweep.")
        return

    ce_n = sum(1 for c in contexts if c["side"] == "CE")
    pe_n = len(contexts) - ce_n
    print(f"Direction split: CE={ce_n}  PE={pe_n}  ({100*ce_n/len(contexts):.1f}% / {100*pe_n/len(contexts):.1f}%)")
    print(f"Entry-frequency: {len(contexts)}/{len(days_covered)} days fire ({100*len(contexts)/len(days_covered):.1f}%)")
    if len(contexts) / len(days_covered) > 0.95:
        print("  *** WARNING: firing on >95% of days -- may have degenerated into something looser than intended. ***")

    print(f"\n################ STOP/TRAIL GRID SWEEP ################")
    results = []
    for stop_pts in STOP_GRID:
        for trail_pts in TRAIL_GRID:
            trades_df = simulate_combo(contexts, stop_pts, trail_pts)
            pf, sqn, n = compute_pf_sqn(trades_df)
            net = trades_df["rupee_pnl"].sum() if n else 0.0
            win = 100 * (trades_df["rupee_pnl"] > 0).mean() if n else 0.0
            trail_label = "none" if trail_pts == 0 else trail_pts
            print(f"  stop={stop_pts:<5} trail={str(trail_label):<5} n={n:<5} win%={win:5.1f}  "
                  f"PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")
            results.append({"stop": stop_pts, "trail": trail_pts, "pf": pf, "sqn": sqn, "n": n, "net": net})

    res_df = pd.DataFrame(results)
    best = res_df.sort_values("sqn", ascending=False).iloc[0]
    print(f"\nBest by SQN: stop={best['stop']} trail={best['trail']}  SQN={best['sqn']:.2f}  "
          f"PF={best['pf']:.2f}  net=Rs{best['net']:,.2f}")
    at_edge = (best["stop"] in (STOP_GRID[0], STOP_GRID[-1])) or (best["trail"] in (TRAIL_GRID[0], TRAIL_GRID[-1]))
    if at_edge:
        print("  *** GRID-EDGE WARNING: best combo sits at the edge of the tested range -- treat as needing "
              "dedicated stress-testing before trusting, not a finding. ***")
    if best["pf"] > 2.5 or (not pd.isna(best["sqn"]) and best["sqn"] > 5):
        print("  *** RED FLAG: PF>2.5 or SQN>5 -- treat as needing a bug hunt before trusting. ***")

    print(f"\n################ DEFAULT COMBO (stop={DEFAULT_STOP}, trail={DEFAULT_TRAIL}) ################")
    default_trades = simulate_combo(contexts, DEFAULT_STOP, DEFAULT_TRAIL)
    default_trades = default_trades.sort_values("entry_time").reset_index(drop=True)
    default_trades.to_csv(TRADES_CSV, index=False)
    pf, sqn, n = compute_pf_sqn(default_trades)
    net = default_trades["rupee_pnl"].sum()
    win = 100 * (default_trades["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{net:,.2f}")

    dates = pd.to_datetime(default_trades["date"])
    if len(dates.unique()) >= 2:
        mid = dates.min() + (dates.max() - dates.min()) / 2
        h1 = default_trades[dates <= mid]
        h2 = default_trades[dates > mid]
        pf1, sqn1, n1 = compute_pf_sqn(h1)
        pf2, sqn2, n2 = compute_pf_sqn(h2)
        print(f"\nOOS split: H1 n={n1} PF={pf1:.2f} SQN={sqn1:.2f} net=Rs{h1['rupee_pnl'].sum():,.2f}")
        print(f"           H2 n={n2} PF={pf2:.2f} SQN={sqn2:.2f} net=Rs{h2['rupee_pnl'].sum():,.2f}")

    equity_curve = STARTING_CAPITAL + default_trades["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")
    if abs(worst_dd) >= STARTING_CAPITAL:
        print("  *** CAPITAL ADEQUACY WARNING. ***")

    daily_pnl = build_daily_pnl(default_trades.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(default_trades, STARTING_CAPITAL)
    print(f"\n################ FULL INSTITUTIONAL METRICS (default combo) ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (full-time): {m['sharpe_fulltime']}   "
          f"Sortino (full-time): {m['sortino_fulltime']}   Max DD: {m['max_dd_pct']}%")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo (5000 reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")
    print(f"Final equity: Rs{m['final_equity']:,.2f}")

    print(f"\nSaved: {TRADES_CSV} ({len(default_trades)} rows)")
    return {"trades": default_trades, "sweep": res_df, "metrics": m}


def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~35 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        df1m = base.fetch_index_chunked(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). Re-run on the machine with a working "
              "Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    run_full_pipeline(df1m)
    print("\nSmoke test complete -- no crashes, all self-checks passed, full sweep pipeline ran "
          "end-to-end on real data.")


def main():
    run_self_checks()
    print("=== NIFTY STATISTICAL PROJECTION ENGINE (MVP) -- 6-MONTH BACKTEST ===")
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
