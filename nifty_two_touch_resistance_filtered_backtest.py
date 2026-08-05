"""
TWO-TOUCH RESISTANCE BREAKOUT, FILTERED -- Monday/Wednesday + 13:15-13:45 excluded
=================================================================
Follow-up to the 2-year sweep result (default 8/4 combo: PF 1.19, SQN
0.83, CAGR 9.49%, p=0.36 -- NOT significant). Two findings from that run
were cross-validated across BOTH the 6-month and 2-year samples
independently (not just a single-sample cut): the 13:15-13:45 session
was the worst performer in both (PF 0.00 in each), and on the larger
2-year sample Monday (PF 0.88, net -Rs717) and Wednesday (PF 0.68, net
-Rs1,797) both looked genuinely weak.

This script excludes those three (Monday, Wednesday, 13:15-13:45
session) and reruns the ORIGINALLY-SPECIFIED 8pt-stop/4pt-trail combo --
NOT the grid-sweep "best" combo (stop=16/trail=2), which was already
flagged as the same tail-risk-harvesting artifact seen twice before in
this project and should not be trusted regardless of how it scores.

HONEST CAVEAT, stated plainly: this is now several cuts deep on the SAME
2-year sample (base result, stop/trail grid, session breakdown, day-of-
week breakdown, and now this exclusion filter). Even if the filtered
result looks better, that alone does not make it a proven edge -- it
needs to hold up on FRESH out-of-sample data collected after today,
not just a rerun of the same historical window sliced a few more times.
Report this filtered result as a hypothesis worth continuing to monitor,
not as a validated strategy.

Entry-signal detection and the per-day context are reused verbatim from
nifty_two_touch_resistance_breakout_backtest.py and nifty_two_touch_
resistance_2yr_sweep.py -- only a post-hoc filter is added here.

Run:
    cd "C:\\dhan algo"
    python nifty_two_touch_resistance_filtered_backtest.py --smoke-test
    python nifty_two_touch_resistance_filtered_backtest.py
"""

import sys
import datetime as dt
import pandas as pd

import nifty_two_touch_resistance_breakout_backtest as base
import nifty_two_touch_resistance_2yr_sweep as sweep
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

FROM_DATE = sweep.FROM_DATE
TO_DATE = sweep.TO_DATE
DEFAULT_STOP, DEFAULT_TRAIL = sweep.DEFAULT_STOP, sweep.DEFAULT_TRAIL  # 8 / 4, your original spec
STARTING_CAPITAL = sweep.STARTING_CAPITAL

EXCLUDED_WEEKDAYS = {0, 2}  # Monday=0, Wednesday=2 (python weekday())
EXCLUDED_SESSION = "13:15-13:45"

TRADES_CSV = "two_touch_resistance_filtered_trades.csv"


def filter_contexts(contexts):
    kept = []
    for ctx in contexts:
        if ctx["day"].weekday() in EXCLUDED_WEEKDAYS:
            continue
        entry_time = ctx["premium_df"].index[ctx["exec_idx"]]
        if base.classify_session(entry_time) == EXCLUDED_SESSION:
            continue
        kept.append(ctx)
    return kept


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    sweep.run_self_checks()

    # --- filter must drop Monday/Wednesday entries and the 13:15-13:45 session,
    # keeping everything else ---
    mon = pd.Timestamp("2025-01-06 09:20")   # Monday
    tue = pd.Timestamp("2025-01-07 09:20")   # Tuesday -- kept
    wed = pd.Timestamp("2025-01-08 13:20")   # Wednesday AND excluded session -- doubly excluded
    thu_bad_session = pd.Timestamp("2025-01-09 13:20")  # Thursday, but excluded session -- dropped
    fri_ok = pd.Timestamp("2025-01-10 10:20")  # Friday, ok session -- kept

    def fake_ctx(ts):
        idx = pd.date_range(ts, periods=1, freq="1min")
        df = pd.DataFrame({"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]}, index=idx)
        return {"day": ts.date(), "premium_df": df, "exec_idx": 0, "resistance": 100.0, "strike": 24000, "lot": 65}

    cands = [fake_ctx(mon), fake_ctx(tue), fake_ctx(wed), fake_ctx(thu_bad_session), fake_ctx(fri_ok)]
    kept = filter_contexts(cands)
    kept_days = {c["day"] for c in kept}
    assert kept_days == {tue.date(), fri_ok.date()}, f"expected only Tuesday and Friday(good session) kept, got {kept_days}"
    print("  filter correctly drops Monday, Wednesday, and the 13:15-13:45 session, keeping the rest ok")

    print("All self-checks passed.\n")


def run_full_pipeline(df1m):
    print("\nPrecomputing entry signals once (2-year sample)...")
    contexts = sweep.precompute_day_contexts(df1m)
    print(f"{len(contexts)} days with a completed entry signal (before filtering).")

    filtered = filter_contexts(contexts)
    dropped = len(contexts) - len(filtered)
    print(f"{dropped} signals dropped (Monday/Wednesday or the 13:15-13:45 session), "
          f"{len(filtered)} remain.")
    if not filtered:
        print("Nothing left after filtering.")
        return

    trades_df = sweep.simulate_combo(filtered, DEFAULT_STOP, DEFAULT_TRAIL)
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(TRADES_CSV, index=False)

    pf, sqn, n = sweep.compute_pf_sqn(trades_df)
    net = trades_df["rupee_pnl"].sum()
    win = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"\n################ FILTERED RESULT (Mon/Wed + 13:15-13:45 excluded, stop=8/trail=4) ################")
    print(f"Trades: {n}   Win rate: {win:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{net:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG: PF>2.5 or SQN>5 -- treat as needing a bug hunt before trusting. ***")

    print(f"\n-- Comparison to the unfiltered 2-year default combo --")
    print(f"  Unfiltered: n=228  PF=1.19  SQN=0.83  net=Rs4,967.45  CAGR=9.49%  p=0.3605 (not significant)")
    print(f"  Filtered:   n={n}  PF={pf:.2f}  SQN={sqn:.2f}  net=Rs{net:,.2f}")

    # OOS split
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    dates = sorted(trades_df["date"].unique())
    if len(dates) >= 2:
        mid = dates[len(dates) // 2]
        h1 = trades_df[trades_df["date"] < mid]
        h2 = trades_df[trades_df["date"] >= mid]
        pf1, sqn1, n1 = sweep.compute_pf_sqn(h1)
        pf2, sqn2, n2 = sweep.compute_pf_sqn(h2)
        print(f"\nOOS split: H1 ({dates[0].date()} to {mid.date()}) n={n1} PF={pf1:.2f} SQN={sqn1:.2f} "
              f"net=Rs{h1['rupee_pnl'].sum():,.2f}")
        print(f"           H2 ({mid.date()} to {dates[-1].date()}) n={n2} PF={pf2:.2f} SQN={sqn2:.2f} "
              f"net=Rs{h2['rupee_pnl'].sum():,.2f}")

    # remaining day/session breakdown -- confirm no single bucket now dominates unexpectedly
    print(f"\n-- Remaining day-of-week breakdown (Mon/Wed already excluded) --")
    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    for wd in ["Tuesday", "Thursday", "Friday"]:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g):
            sweep.print_combo_row(wd, g)

    equity_curve = STARTING_CAPITAL + trades_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")

    daily_pnl = build_daily_pnl(trades_df.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(trades_df, STARTING_CAPITAL)
    print(f"\n################ FILTERED -- FULL INSTITUTIONAL METRICS ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (full-time): {m['sharpe_fulltime']}   "
          f"Sortino (full-time): {m['sortino_fulltime']}   Max DD: {m['max_dd_pct']}%")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo (5000 reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")

    print(f"\n*** REMINDER: this is several cuts deep on the SAME 2-year sample (base result, stop/trail "
          f"grid, session breakdown, day-of-week breakdown, and now this exclusion filter). Even a clean "
          f"result here is a hypothesis, not a validated edge -- it needs to hold up on fresh data collected "
          f"AFTER today before trusting it with real capital. ***")

    return {"trades": trades_df, "metrics": m, "monte_carlo": mc}


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
    print("\nSmoke test complete -- no crashes, all self-checks passed, full pipeline ran end-to-end "
          "on real data.")


def main():
    run_self_checks()
    print("=== TWO-TOUCH RESISTANCE, FILTERED (Mon/Wed + 13:15-13:45 excluded) -- 2-YEAR BACKTEST ===")
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
