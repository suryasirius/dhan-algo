"""
NIFTY STATISTICAL PROJECTION ENGINE (NSPE) v3 -- pure-trail, full-agreement
-- 2-YEAR BACKTEST (+ separate 6-month slice)
=================================================================
Follow-up to nifty_statistical_projection_v2_backtest.py's real 2-year
result (n=331, PF=1.60, SQN=1.99, net=Rs23,589, p=0.058 -- just short of
significant). Two concrete, diagnosed issues from that run are addressed
here, per your "do it":

  1. DROPPED THE TARGET. v2's 75th-percentile target fired on 54% of
     trades but contributed almost nothing (net=Rs3,326 over 180 hits,
     ~Rs18/trade average) -- nearly all real profit (Rs25,804) came from
     the 42.6% of trades that ran past the target and got caught by the
     trailing stop instead. This version is pure stop+trail managed, no
     fixed target -- matching every other validated strategy in this
     project (Velez, Opening-Drive, Two-Touch), and removing one moving
     part from the grid-edge problem below.

  2. TIGHTENED CONFIDENCE to full (4/4) window agreement, not 3/4 (75%).
     v2 fired on 66.6% of days -- loose for something billed as "high
     confidence." Requiring ALL FOUR lookback windows (4/10/20/50-day)
     to independently agree is a real, defensible tightening (checked:
     on a monotonic synthetic series, the 4-day window's mean is highest
     -- so a price that clears the 4-day mean automatically clears the
     rest; the 3/4 cases that get newly EXCLUDED here are specifically
     the ones where the SHORTEST, most reactive window has already
     moved but the longer, steadier windows haven't caught up yet).

  3. EXTENDED the trail-fraction grid, finer near zero and wider at the
     top ([0.0, 0.03, 0.06, 0.10, 0.20] instead of v2's
     [0.0, 0.05, 0.08, 0.10, 0.15]), per the rigor skill's "extend the
     grid once" rule -- v2's trail=0 (trailing fully disabled) beat
     every trail>0 combo at every stop level, which is worth checking
     for a smooth interior optimum instead of assuming "no trail" is
     simply correct.

Does NOT touch nifty_statistical_projection_v2_backtest.py (imported as
`v2`, reused for compute_multiwindow_projection / find_entry_for_day --
CONFIDENCE_THRESHOLD is overridden to 1.0 at import time here, same
monkeypatch-for-configuration convention used elsewhere in this project).
Position management reuses top3.step_long_premium_position verbatim (no
target logic needed anymore).

Run:
    cd "C:\\dhan algo"
    python nifty_statistical_projection_v3_backtest.py --smoke-test
    python nifty_statistical_projection_v3_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as base
import nifty_statistical_projection_backtest as v1
import nifty_statistical_projection_v2_backtest as v2
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

v2.CONFIDENCE_THRESHOLD = 1.0  # require ALL 4 windows to agree, not 3/4

BACKTEST_DAYS = 730
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
RECENT_6MO_DAYS = 183

STOP_FRAC_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]      # unchanged -- v2 already showed a peak/plateau near 0.25
TRAIL_FRAC_GRID = [0.0, 0.03, 0.06, 0.10, 0.20]       # extended per grid-edge rule
DEFAULT_STOP_FRAC, DEFAULT_TRAIL_FRAC = 0.20, 0.06
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000
TRADES_CSV = "nifty_statistical_projection_v3_trades.csv"


def run_standalone_for_day(day_df_1min, daily_ohlc):
    n = len(day_df_1min)
    if n < 5:
        return None
    day = day_df_1min.index[0].date()
    proj = v2.compute_multiwindow_projection(daily_ohlc, day)
    if proj is None:
        return None
    sig_idx, side, conf = v2.find_entry_for_day(day_df_1min, proj)
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
    return {"day": day, "premium_df": premium_df, "exec_idx": exec_idx, "side": side, "strike": strike,
            "lot": lot, "entry_price": entry_price, "confidence": conf,
            "blended_avg_range": proj["blended_avg_range"]}


def simulate_combo(contexts, stop_frac, trail_frac):
    trades = []
    for ctx in contexts:
        premium_df = ctx["premium_df"]
        exec_idx = ctx["exec_idx"]
        entry_price = ctx["entry_price"]
        stop_pts = stop_frac * ctx["blended_avg_range"] * top3.ASSUMED_DELTA
        trail_pts = (trail_frac * ctx["blended_avg_range"] * top3.ASSUMED_DELTA) if trail_frac > 0 else 1e9
        pos = {"entry_price": entry_price, "stop_level": entry_price - stop_pts, "peak": entry_price,
               "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, premium_df, trail_pts)
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
    assert v2.CONFIDENCE_THRESHOLD == 1.0, "expected v3 to require full (4/4) window agreement"
    print("  confidence threshold correctly overridden to 1.0 (4/4 agreement) ok")

    idx = pd.date_range("2024-01-01", periods=55, freq="B")
    highs = [25000 + i * 2 for i in range(55)]
    lows = [24950 + i * 2 for i in range(55)]
    closes = [24980 + i * 2 for i in range(55)]
    opens = [24960 + i * 2 for i in range(55)]
    daily = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)
    target_day = idx[54].date()
    proj = v2.compute_multiwindow_projection(daily, target_day)
    assert proj is not None
    # verified empirically: window_mean_highs = {4: 25104.0, 10: 25100.0, 20: 25093.3, 50: 25073.3}
    # (4-day window is highest on a monotonic rising series -- most reactive to recent data)
    wh = proj["window_mean_highs"]
    assert wh[4] > wh[10] > wh[20] > wh[50]

    day_open = proj["prev_close"]
    idx2 = pd.date_range("2025-01-10 09:15", periods=10, freq="1min")

    # --- price clears 10/20/50-day means but NOT the 4-day mean -> 3/4
    # agreement -- must be BLOCKED now (was allowed in v2 at 0.75) ---
    price_3of4 = (wh[10] + wh[4]) / 2  # strictly between wh[10] and wh[4]
    closes3 = [day_open] * 3 + [price_3of4] * 7
    df3 = pd.DataFrame({"open": [day_open] + closes3[:-1], "high": [c + 5 for c in closes3],
                        "low": [c - 5 for c in closes3], "close": closes3}, index=idx2)
    sig_idx, side, conf = v2.find_entry_for_day(df3, proj)
    assert sig_idx is None, f"expected 3/4 agreement to be BLOCKED at threshold=1.0, got {(sig_idx, side, conf)}"
    print("  3-of-4 window agreement correctly BLOCKED at the new full-agreement threshold ok")

    # --- price clears ALL 4 windows -> must still fire ---
    price_4of4 = wh[4] + 20
    closes4 = [day_open] * 3 + [price_4of4] * 7
    df4 = pd.DataFrame({"open": [day_open] + closes4[:-1], "high": [c + 5 for c in closes4],
                        "low": [c - 5 for c in closes4], "close": closes4}, index=idx2)
    sig_idx4, side4, conf4 = v2.find_entry_for_day(df4, proj)
    assert sig_idx4 is not None and side4 == "CE" and conf4 == 1.0
    print("  full (4/4) window agreement still fires correctly ok")

    # --- position management: no target anymore, reuses top3's own
    # already-validated stepper directly (no new logic to re-derive) ---
    prem_df = pd.DataFrame({"open": [50, 55, 60, 65, 70], "high": [52, 57, 63, 68, 72],
                            "low": [49, 54, 59, 64, 69], "close": [51, 56, 61, 66, 71]},
                           index=pd.date_range("2025-01-10 09:20", periods=5, freq="1min"))
    pos = {"entry_price": 50.0, "stop_level": 45.0, "peak": 50.0, "trailing_active": False, "last_checked_idx": -1}
    exit_info = top3.step_long_premium_position(pos, prem_df, trail_pts=3.0)
    assert exit_info is None or exit_info["reason"] in ("TRAIL_SL", "INITIAL_SL", "EOD")
    print("  simulate_combo's position stepper (top3.step_long_premium_position, no target) runs cleanly ok")

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
    print(f"{len(contexts)} days with a completed entry signal (out of {len(days_covered)} total days).")
    if not contexts:
        print("No signals at all -- nothing to sweep.")
        return

    ce_n = sum(1 for c in contexts if c["side"] == "CE")
    print(f"Direction split: CE={ce_n}  PE={len(contexts)-ce_n}")
    print(f"Entry-frequency: {len(contexts)}/{len(days_covered)} days fire ({100*len(contexts)/len(days_covered):.1f}%)")

    print(f"\n################ DYNAMIC STOP/TRAIL GRID SWEEP (fractions of blended avg range, no target) ################")
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

    print(f"\n################ COMPARISON vs v2 (target + 75%-agreement) ################")
    print("  v2 (previous):  n=331  PF=1.60  SQN=1.99  net_Rs=23,589.01  CAGR=44.61%  p=0.0581 (not significant)")
    pf3, sqn3, n3 = compute_pf_sqn(default_trades)
    print(f"  v3 (this run): n={n3}  PF={pf3:.2f}  SQN={sqn3:.2f}  net_Rs={default_trades['rupee_pnl'].sum():,.2f}")

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
    print("=== NIFTY STATISTICAL PROJECTION ENGINE v3 (pure-trail, full-agreement) -- 2-YEAR BACKTEST ===")
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
