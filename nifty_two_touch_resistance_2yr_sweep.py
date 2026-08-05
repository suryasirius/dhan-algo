"""
TWO-TOUCH RESISTANCE BREAKOUT -- 2-YEAR EXTENSION + STOP/TRAIL SWEEP
=================================================================
Follow-up to nifty_two_touch_resistance_breakout_backtest.py's 6-month
result (61 trades, PF 1.82, SQN 1.28, p=0.185 -- NOT statistically
significant, and the day/session breakdowns had too few trades per
bucket, n=1-14, to trust).

WHY EXTEND TO 2 YEARS: unlike the M-Structure strategy, this pattern uses
no volume filter at all -- entry is pure price action (opening-range
breach, two-touch pivot resistance, third-breach confirm) -- so, unlike
M-Structure, there's no missing-volume-data blocker. The full 2-year
Nifty index history is usable here, giving roughly 4x the sample and a
real shot at statistical significance and meaningful day/session splits.

Entry-signal detection is reused VERBATIM from nifty_two_touch_
resistance_breakout_backtest.py (import as `base`) and computed ONCE per
day; only the exit simulation (stop distance below resistance, trail
distance) varies across the grid -- same efficient design as the
M-Structure sweep.

Run:
    cd "C:\\dhan algo"
    python nifty_two_touch_resistance_2yr_sweep.py --smoke-test
    python nifty_two_touch_resistance_2yr_sweep.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as base
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")

STOP_GRID = [4.0, 8.0, 12.0, 16.0]   # points below the resistance level
TRAIL_GRID = [2.0, 4.0, 6.0, 8.0]
DEFAULT_STOP, DEFAULT_TRAIL = base.STOP_BELOW_RESISTANCE_PTS, base.TRAIL_PTS  # 8 / 4, your original spec

BROKERAGE_PER_TRADE = base.BROKERAGE_PER_TRADE
STARTING_CAPITAL = base.STARTING_CAPITAL


# =============================================================================
# Precompute signals once per day (independent of stop/trail), reuse across combos
# =============================================================================
def precompute_day_contexts(df1m):
    days_covered = sorted(set(df1m.index.date))
    contexts = []
    for day in days_covered:
        day_df = df1m[df1m.index.date == day]
        if len(day_df) < base.OPENING_RANGE_BARS + 10:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike = base.atm_strike(day_open_spot)
        premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="CE")
        sig_bar, resistance, t1, t2 = base.find_two_touch_resistance_and_entry(premium_df)
        if sig_bar is None:
            continue
        exec_idx = sig_bar + 1
        if exec_idx >= len(day_df):
            continue
        lot = top3.lot_size_for_date(day)
        contexts.append({
            "day": day, "premium_df": premium_df, "resistance": resistance,
            "exec_idx": exec_idx, "strike": strike, "lot": lot,
        })
    return contexts


def simulate_combo(contexts, stop_below_resistance_pts, trail_pts):
    trades = []
    for ctx in contexts:
        premium_df = ctx["premium_df"]
        exec_idx = ctx["exec_idx"]
        entry_price = premium_df["open"].iloc[exec_idx]
        pos = {"entry_price": entry_price, "stop_level": ctx["resistance"] - stop_below_resistance_pts,
               "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, premium_df, trail_pts)
        if exit_info is None:
            exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": premium_df.index[-1], "exit_idx": len(premium_df) - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        rupee_pnl = pnl_pts * ctx["lot"] - BROKERAGE_PER_TRADE
        trades.append({
            "side": "CE", "strike": ctx["strike"], "resistance": ctx["resistance"],
            "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
            "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
            "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": ctx["day"],
        })
    return pd.DataFrame(trades)


def compute_pf_sqn(trades_df):
    pnl = trades_df["rupee_pnl"]
    n = len(pnl)
    if n == 0:
        return 0.0, 0.0, 0
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    std = pnl.std(ddof=1) if n > 1 else 0.0
    sqn = (pnl.mean() / std * math.sqrt(n)) if std > 0 else 0.0
    return pf, sqn, n


def print_combo_row(label, g):
    pf, sqn, n = compute_pf_sqn(g)
    net = g["rupee_pnl"].sum() if n else 0.0
    win = 100 * (g["rupee_pnl"] > 0).mean() if n else 0.0
    print(f"  {label:<22} n={n:<5} win%={win:5.1f}  PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")
    return pf, sqn, n, net


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    base.run_self_checks()

    # --- equivalence check: simulate_combo(default 8/4) must match the
    # original script's own hand-traced example exactly ---
    idx = pd.date_range("2025-01-06 09:15", periods=50, freq="1min")
    closes = [100.0] * 30
    closes += [105, 108, 112, 110, 108]
    closes += [105, 104, 106, 111, 109, 107]
    closes += [109, 113]
    closes += [114, 116, 118, 120, 122, 124, 126, 128]
    closes = closes[:50]
    df = pd.DataFrame({
        "open": [100.0] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)
    sig_bar, resistance, t1, t2 = base.find_two_touch_resistance_and_entry(df)
    exec_idx = sig_bar + 1
    lot = top3.lot_size_for_date(idx[0].date())
    ctx = [{"day": idx[0].date(), "premium_df": df, "resistance": resistance,
            "exec_idx": exec_idx, "strike": base.atm_strike(df["open"].iloc[0]), "lot": lot}]
    swept = simulate_combo(ctx, DEFAULT_STOP, DEFAULT_TRAIL)
    assert len(swept) == 1
    entry_price = df["open"].iloc[exec_idx]
    pos_direct = {"entry_price": entry_price, "stop_level": resistance - DEFAULT_STOP, "peak": entry_price,
                  "trailing_active": False, "last_checked_idx": exec_idx - 1}
    exit_direct = top3.step_long_premium_position(pos_direct, df, DEFAULT_TRAIL)
    if exit_direct is None:
        # position still open at data end (this synthetic day trends up the whole time with no EOD
        # marker in range) -- simulate_combo must fall back to the same EOD_FALLBACK convention.
        assert swept.iloc[0]["reason"] == "EOD_FALLBACK"
        assert abs(swept.iloc[0]["exit_price"] - df["close"].iloc[-1]) < 1e-9
    else:
        assert abs(swept.iloc[0]["exit_price"] - exit_direct["exit_price"]) < 1e-9, (
            f"sweep engine diverged from direct computation: {swept.iloc[0]['exit_price']} vs {exit_direct['exit_price']}"
        )
    print("  simulate_combo(default 8/4) matches direct computation exactly (open-at-data-end handled consistently) ok")

    print("All self-checks passed.\n")


def run_full_pipeline(df1m):
    print("\nPrecomputing entry signals once (reused across every stop/trail combo)...")
    contexts = precompute_day_contexts(df1m)
    print(f"{len(contexts)} days with a completed entry signal (out of {len(set(df1m.index.date))} total days).")
    if not contexts:
        print("No signals at all -- nothing to sweep.")
        return

    print(f"\n################ STOP(below resistance)/TRAIL GRID SWEEP ################")
    results = []
    for stop_pts in STOP_GRID:
        for trail_pts in TRAIL_GRID:
            trades_df = simulate_combo(contexts, stop_pts, trail_pts)
            pf, sqn, n, net = print_combo_row(f"stop={stop_pts} trail={trail_pts}", trades_df)
            results.append({"stop": stop_pts, "trail": trail_pts, "pf": pf, "sqn": sqn, "n": n, "net": net})

    res_df = pd.DataFrame(results)
    best = res_df.sort_values("sqn", ascending=False).iloc[0]
    print(f"\nBest by SQN: stop={best['stop']} trail={best['trail']}  SQN={best['sqn']:.2f}  "
          f"PF={best['pf']:.2f}  net=Rs{best['net']:,.2f}")
    at_edge = (best["stop"] in (STOP_GRID[0], STOP_GRID[-1])) or (best["trail"] in (TRAIL_GRID[0], TRAIL_GRID[-1]))
    if at_edge:
        print("  *** GRID-EDGE WARNING per project rigor rule: best combo sits at the edge of the tested "
              "range -- treat as needing dedicated stress-testing before trusting, not a finding. Your "
              "originally-specified 8/4 combo has an actual basis; the grid-corner combo may not. ***")
    if best["pf"] > 2.5 or best["sqn"] > 5:
        print("  *** RED FLAG: PF>2.5 or SQN>5 -- treat as needing a bug hunt before trusting. ***")

    print(f"\n################ DEFAULT COMBO (stop={DEFAULT_STOP}, trail={DEFAULT_TRAIL}, your original spec) ################")
    default_trades = simulate_combo(contexts, DEFAULT_STOP, DEFAULT_TRAIL)
    pf, sqn, n, net = print_combo_row("headline", default_trades)

    # in-sample / out-of-sample split on the default combo
    default_trades["date"] = pd.to_datetime(default_trades["date"])
    dates = sorted(default_trades["date"].unique())
    if len(dates) >= 2:
        mid = dates[len(dates) // 2]
        h1 = default_trades[default_trades["date"] < mid]
        h2 = default_trades[default_trades["date"] >= mid]
        print(f"\nOOS split: H1 ({dates[0].date()} to {mid.date()})")
        print_combo_row("  H1", h1)
        print(f"           H2 ({mid.date()} to {dates[-1].date()})")
        print_combo_row("  H2", h2)

    print(f"\n################ 30-MINUTE SESSION-OF-DAY BREAKDOWN (default combo, 2yr sample) ################")
    default_trades["session"] = default_trades["entry_time"].apply(base.classify_session)
    for sess in base.SESSIONS:
        g = default_trades[default_trades["session"] == sess]
        if len(g):
            print_combo_row(sess, g)

    print(f"\n################ DAY-OF-WEEK BREAKDOWN (default combo, 2yr sample) ################")
    default_trades["weekday"] = default_trades["entry_time"].dt.day_name()
    for wd in base.WEEKDAY_ORDER:
        g = default_trades[default_trades["weekday"] == wd]
        if len(g):
            print_combo_row(wd, g)

    default_trades.to_csv("two_touch_resistance_2yr_default_combo_trades.csv", index=False)

    equity_curve = STARTING_CAPITAL + default_trades.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")

    daily_pnl = build_daily_pnl(default_trades.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(default_trades, STARTING_CAPITAL)
    print(f"\n################ DEFAULT COMBO -- FULL INSTITUTIONAL METRICS (2yr sample) ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (full-time): {m['sharpe_fulltime']}   "
          f"Sortino (full-time): {m['sortino_fulltime']}   Max DD: {m['max_dd_pct']}%")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo (5000 reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")


def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~35 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        df1m = base.fetch_index_chunked(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). This failure is specifically about "
              "reaching the Dhan API from wherever this was just run -- re-run on the machine with a "
              "working Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    run_full_pipeline(df1m)
    print("\nSmoke test complete -- no crashes, all self-checks passed, full sweep pipeline ran "
          "end-to-end on real data.")


def main():
    run_self_checks()
    print("=== TWO-TOUCH RESISTANCE -- 2-YEAR EXTENSION + STOP/TRAIL SWEEP ===")
    print("Fetching Nifty index 1-min data (2 years)...")
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
