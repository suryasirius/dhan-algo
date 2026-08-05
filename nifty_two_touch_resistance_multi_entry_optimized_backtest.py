"""
TWO-TOUCH RESISTANCE MULTI-ENTRY, OPTIMIZED -- cap to first 2 entries/day +
Wednesday/13:15-13:45 exclusion -- 2-YEAR BACKTEST
=================================================================
Follow-up to the real 2-year multi-entry result (n=473, PF=1.19, SQN=1.16,
net=Rs10,096.34, p=0.2045 -- NOT significant, maxDD -23.4%). Breaking that
result down by entry-number-within-the-day and by weekday showed two
real, sample-backed patterns worth acting on:

  1. ENTRY #1 + #2 combined: 362 trades, net=Rs11,021.61 (net=Rs4,967.45 +
     Rs6,054.16) -- genuinely solid. ENTRY #3 onward: 111 trades, net=
     -Rs925.27 -- a net drag, with entry #3 alone (n=70) a clean loser.
     Entries #5-6 are too thin (n=12, n=3) to trust either way. This
     script caps the day to the first 2 re-arms only (MAX_ENTRIES_KEPT=2)
     and drops anything beyond that.

  2. Wednesday: PF=0.82, net=-Rs2,171.18 (n=115) -- a clear loser, and NOT
     a fresh cut of this data: it re-confirms something already validated
     independently in nifty_two_touch_resistance_filtered_backtest.py on
     the single-entry version. The 13:15-13:45 session was ALSO already
     validated as the worst session in that same prior work, and shows up
     mildly negative here too (PF=0.97, net=-Rs150.33). Both are applied
     here as a REUSED, already-proven filter -- not an invented one.

  NOT applied: Monday. Unlike the single-entry study (where Monday was
  also weak), Monday in THIS multi-entry run is roughly flat (PF=1.00,
  net=Rs47.64) -- not a loser, so excluding it here would be a fresh,
  unsupported cut on this exact dataset. Left in.

  NOT touched: the 8pt/4pt stop-trail distances, and the newly-observed
  10:45-11:45 "dead zone" session block seen in the raw multi-entry run.
  Both are explicitly flagged as things NOT to act on yet -- widening/
  tightening stops based on this same sample is the exact grid-edge/
  tail-risk trap already found once on Opening-Drive in this project, and
  the 10:45-11:45 weakness is a brand-new cut of this data with no prior
  independent confirmation, so it stays a hypothesis, not a rule.

Does NOT touch nifty_two_touch_resistance_multi_entry_backtest.py,
nifty_two_touch_resistance_breakout_backtest.py, or
nifty_two_touch_resistance_filtered_backtest.py (all locked/already
validated). Imports the multi-entry day-runner verbatim and applies a
post-hoc filter on its output -- same "filter, don't rewrite entry logic"
approach as the original filtered script.

HONEST CAVEAT: this is now several cuts deep on the multi-entry sample
(raw multi-entry result, entry-number breakdown, weekday breakdown,
session breakdown, and now this filter). Even a clean number here is a
hypothesis worth continuing to monitor on FRESH data collected after
today, not a validated edge to trade on size immediately.

Run:
    cd "C:\\dhan algo"
    python nifty_two_touch_resistance_multi_entry_optimized_backtest.py --smoke-test
    python nifty_two_touch_resistance_multi_entry_optimized_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_two_touch_resistance_breakout_backtest as base
import nifty_two_touch_resistance_multi_entry_backtest as multi
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

FROM_DATE = multi.FROM_DATE
TO_DATE = multi.TO_DATE
STARTING_CAPITAL = multi.STARTING_CAPITAL

MAX_ENTRIES_KEPT = 2                # keep only entry_num_today <= 2
EXCLUDED_WEEKDAYS = {2}             # Wednesday only (python weekday(): Mon=0 ... Wed=2). Monday NOT excluded here.
EXCLUDED_SESSION = "13:15-13:45"

TRADES_CSV = "two_touch_resistance_multi_entry_optimized_trades.csv"


def apply_filters(trades_df):
    if trades_df is None or trades_df.empty:
        return trades_df
    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    keep = trades_df["entry_num_today"] <= MAX_ENTRIES_KEPT
    keep &= ~trades_df["entry_time"].dt.weekday.isin(EXCLUDED_WEEKDAYS)
    keep &= trades_df["entry_time"].apply(lambda t: base.classify_session(t) != EXCLUDED_SESSION)
    return trades_df[keep].reset_index(drop=True)


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    multi.run_self_checks()
    print("  multi.run_self_checks() re-verified ok (multi-entry re-arm primitives)")

    # --- unit test the filter directly on a small synthetic trades table
    # covering every drop/keep case ---
    rows = [
        # kept: entry #1, Tuesday, good session
        {"entry_num_today": 1, "entry_time": pd.Timestamp("2025-01-07 09:50"), "rupee_pnl": 100.0},
        # kept: entry #2, Friday, good session
        {"entry_num_today": 2, "entry_time": pd.Timestamp("2025-01-10 10:20"), "rupee_pnl": 50.0},
        # dropped: entry #3 (beyond cap), otherwise fine
        {"entry_num_today": 3, "entry_time": pd.Timestamp("2025-01-07 10:20"), "rupee_pnl": -30.0},
        # dropped: Wednesday, entry #1, good session
        {"entry_num_today": 1, "entry_time": pd.Timestamp("2025-01-08 10:20"), "rupee_pnl": 40.0},
        # dropped: excluded session (13:15-13:45), entry #1, Thursday
        {"entry_num_today": 1, "entry_time": pd.Timestamp("2025-01-09 13:20"), "rupee_pnl": -10.0},
        # kept: Monday is explicitly NOT excluded here, entry #2, good session
        {"entry_num_today": 2, "entry_time": pd.Timestamp("2025-01-06 11:00"), "rupee_pnl": 20.0},
    ]
    df = pd.DataFrame(rows)
    filtered = apply_filters(df)
    assert len(filtered) == 3, f"expected 3 rows kept, got {len(filtered)}"
    assert set(filtered["rupee_pnl"]) == {100.0, 50.0, 20.0}, (
        f"unexpected rows kept: {filtered['rupee_pnl'].tolist()}")
    print("  filter unit test: caps to entry #1-#2, drops Wednesday, drops 13:15-13:45 session, "
          "keeps Monday ok")

    # --- integration check: the multi-entry day-runner's own hand-verified
    # two-pattern synthetic day (from multi.run_self_checks) should, once
    # filtered here, keep BOTH trades (only 2 exist, both <= cap; the day
    # used there is a Monday, not excluded by this filter) ---
    def _pattern_block(base_level):
        vals = [base_level] * 30
        vals += [base_level + 5, base_level + 8, base_level + 12, base_level + 10, base_level + 8]
        vals += [base_level + 5, base_level + 4, base_level + 6, base_level + 11, base_level + 9, base_level + 7]
        vals += [base_level + 9, base_level + 13]
        vals += [base_level + 14, base_level + 16, base_level + 18, base_level + 20]
        return vals

    block1 = _pattern_block(100.0)
    pullback = [116.0, 112.0, 111.0]
    block2 = [119.0, 118.0, 120.0, 119.0] + _pattern_block(120.0)[30:]
    closes = block1 + pullback + block2
    idx = pd.date_range("2025-01-06 09:15", periods=len(closes), freq="1min")  # 2025-01-06 is a Monday
    day_df = pd.DataFrame({
        "open": [closes[0]] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)
    # run_standalone_for_day_multi expects a raw INDEX day_df (it builds its own
    # premium series internally via top3.build_premium_series(side="CE")), but
    # our hand-crafted day_df here IS already a premium-shaped series (same
    # convention as base's own self-check). Feed it through the same direct
    # loop multi.run_self_checks() already validated instead of going through
    # build_premium_series again, then filter that output.
    import dhan_live_bot as dlb
    import nifty_paper_trade_top3_live as top3
    pivot_series = dlb.pivot_high(day_df["high"], base.PIVOT_LEFT, base.PIVOT_RIGHT)
    highs = day_df["high"].to_numpy()
    reference_high = highs[:base.OPENING_RANGE_BARS].max()
    search_from = base.OPENING_RANGE_BARS
    manual_trades = []
    while len(manual_trades) < multi.MAX_TRADES_PER_DAY:
        result = multi._find_next_signal(day_df, pivot_series, reference_high, search_from)
        if result is None:
            break
        sig_bar, resistance, t1, t2 = result
        exec_idx = sig_bar + 1
        if exec_idx >= len(day_df):
            break
        entry_price = day_df["open"].iloc[exec_idx]
        pos = {"entry_price": entry_price, "stop_level": resistance - base.STOP_BELOW_RESISTANCE_PTS,
               "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, day_df, base.TRAIL_PTS)
        if exit_info is None:
            exit_info = {"exit_price": day_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": day_df.index[-1], "exit_idx": len(day_df) - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        manual_trades.append({
            "entry_num_today": len(manual_trades) + 1, "entry_time": day_df.index[exec_idx],
            "exit_time": exit_info["exit_time"], "rupee_pnl": pnl_pts * 65 - base.BROKERAGE_PER_TRADE,
            "reason": exit_info["reason"],
        })
        exit_idx = exit_info.get("exit_idx", len(day_df) - 1)
        if exit_idx >= len(day_df) - 1:
            break
        reference_high = max(reference_high, highs[:exit_idx + 1].max())
        search_from = exit_idx + 1

    assert len(manual_trades) == 2
    trades_df2 = pd.DataFrame(manual_trades)
    filtered2 = apply_filters(trades_df2)
    assert len(filtered2) == 2, f"expected both trades kept (Monday, entries 1-2, cap=2), got {len(filtered2)}"
    print("  integration check: multi-entry's own 2-trade synthetic day survives the filter intact "
          "(Monday kept, both within the entry-count cap) ok")

    print("All self-checks passed.\n")


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


def print_combo_row(label, g):
    pf, sqn, n = compute_pf_sqn(g)
    net = g["rupee_pnl"].sum() if n else 0.0
    win = 100 * (g["rupee_pnl"] > 0).mean() if n else 0.0
    print(f"  {label:<20} n={n:<5} win%={win:5.1f}  PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")


def run_full_pipeline(df1m, label="2-YEAR OPTIMIZED MULTI-ENTRY BACKTEST"):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        all_trades.extend(multi.run_standalone_for_day_multi(day_df))
        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} raw trades so far")

    if not all_trades:
        print("No trades generated at all.")
        return None

    raw_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    filtered_df = apply_filters(raw_df)
    dropped = len(raw_df) - len(filtered_df)
    print(f"\n{len(raw_df)} raw multi-entry trades -> {dropped} dropped (entry #3+, Wednesday, or "
          f"{EXCLUDED_SESSION}) -> {len(filtered_df)} remain.")

    if filtered_df.empty:
        print("Nothing left after filtering.")
        return None

    filtered_df.to_csv(TRADES_CSV, index=False)
    print(f"{len(filtered_df)} trades saved to {TRADES_CSV}")

    pf, sqn, n = compute_pf_sqn(filtered_df)
    net = filtered_df["rupee_pnl"].sum()
    win = 100 * (filtered_df["rupee_pnl"] > 0).mean()
    print(f"\n################ OPTIMIZED MULTI-ENTRY -- HEADLINE RESULT ################")
    print(f"Trades: {n}   Win rate: {win:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{net:,.2f}")
    if pf > 2.5 or (not math.isnan(sqn) and sqn > 5):
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt "
              "before trusting this. ***")

    print(f"\n-- Stop-loss usage after optimization --")
    for reason, g in filtered_df.groupby("reason"):
        print(f"  {reason:<14} n={len(g):<5} ({100*len(g)/len(filtered_df):5.1f}%)  net=Rs{g['rupee_pnl'].sum():>10,.2f}")

    filtered_df["date"] = pd.to_datetime(filtered_df["date"]) if "date" in filtered_df.columns else filtered_df["entry_time"].dt.normalize()
    dates = sorted(filtered_df["date"].unique())
    if len(dates) >= 2:
        mid = dates[len(dates) // 2]
        h1 = filtered_df[filtered_df["date"] < mid]
        h2 = filtered_df[filtered_df["date"] >= mid]
        pf1, sqn1, n1 = compute_pf_sqn(h1)
        pf2, sqn2, n2 = compute_pf_sqn(h2)
        print(f"\nOOS split: H1 ({dates[0]} to {mid}) n={n1} PF={pf1:.2f} SQN={sqn1:.2f} net=Rs{h1['rupee_pnl'].sum():,.2f}")
        print(f"           H2 ({mid} to {dates[-1]}) n={n2} PF={pf2:.2f} SQN={sqn2:.2f} net=Rs{h2['rupee_pnl'].sum():,.2f}")

    print(f"\n-- Remaining weekday breakdown (Wednesday already excluded) --")
    filtered_df["weekday"] = filtered_df["entry_time"].dt.day_name()
    for wd in ["Monday", "Tuesday", "Thursday", "Friday"]:
        g = filtered_df[filtered_df["weekday"] == wd]
        if len(g):
            print_combo_row(wd, g)

    equity_curve = STARTING_CAPITAL + filtered_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")

    daily_pnl = build_daily_pnl(filtered_df.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(filtered_df, STARTING_CAPITAL)
    print(f"\n################ OPTIMIZED -- FULL INSTITUTIONAL METRICS ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (full-time): {m['sharpe_fulltime']}   "
          f"Sortino (full-time): {m['sortino_fulltime']}   Max DD: {m['max_dd_pct']}%")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo (5000 reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")
    print(f"Final equity: Rs{m['final_equity']:,.2f}")

    print(f"\n################ COMPARISON ################")
    print("  Single-entry (original):     n=228  PF=1.19  SQN=0.83  net_Rs=4,967.45   CAGR=9.49%   p=0.36 (not significant)")
    print("  Raw multi-entry (unfiltered): n=473  PF=1.19  SQN=1.16  net_Rs=10,096.34  CAGR=18.5%   p=0.2045 (not significant)")
    print(f"  Optimized multi-entry:       n={n}  PF={pf:.2f}  SQN={sqn:.2f}  net_Rs={net:,.2f}  "
          f"CAGR={m['cagr_pct']}%   p={m['p_value']} ({'significant' if m['p_value'] < 0.05 else 'not significant'})")

    print(f"\n*** REMINDER: several cuts deep on the multi-entry sample already (raw result, entry-number "
          f"breakdown, weekday breakdown, session breakdown, and now this filter). Treat as a hypothesis "
          f"needing fresh out-of-sample confirmation, not a validated edge. ***")

    return {"trades": filtered_df, "metrics": m, "monte_carlo": mc}


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
    run_full_pipeline(df1m, label="SMOKE TEST (recent ~35 calendar days)")
    print("\nSmoke test complete -- no crashes, all self-checks passed, full optimized pipeline ran "
          "end-to-end on real data.")


def main():
    run_self_checks()
    print("=== TWO-TOUCH RESISTANCE MULTI-ENTRY, OPTIMIZED -- 2-YEAR REAL BACKTEST ===")
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
